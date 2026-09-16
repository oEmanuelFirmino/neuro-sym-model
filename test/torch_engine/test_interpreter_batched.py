"""Teste de ponte: constrói o MESMO experimento nos dois motores (embeddings e
pesos copiados do motor antigo pro novo, não apenas mesma seed -- RNGs diferentes
não dão valores bit-idênticos), roda B=1 nos dois e compara a saída. Isto prova
que `BatchedInterpreter._eval` é semanticamente idêntico a
`Interpreter.eval_formula`, não apenas "parece igual"."""

import pytest

torch = pytest.importorskip("torch")

from src.neurosym.interpreter import Interpreter
from src.neurosym.logic import Atom, Constant, Implies, Not
from src.neurosym.module.module import Linear as OldLinear
from src.neurosym.module.module import ReLU as OldReLU
from src.neurosym.module.module import Sequential as OldSequential
from src.neurosym.module.module import Sigmoid as OldSigmoid
from src.neurosym.tensor import Tensor
from src.neurosym.tensor.backend import set_backend
from src.neurosym.torch_engine import BatchedGroundingEnv, BatchedInterpreter, build_predicate_mlp
from src.neurosym.torch_engine.compile import compile_formulas

EMBEDDING_DIM = 4
HIDDEN = 8
ENTITIES = ["a", "b", "c"]


def _copy_linear_stack(old_seq: OldSequential, new_seq: torch.nn.Sequential) -> None:
    old_linears = [m for m in old_seq._modules.values() if isinstance(m, OldLinear)]
    new_linears = [m for m in new_seq if isinstance(m, torch.nn.Linear)]
    assert len(old_linears) == len(new_linears)
    with torch.no_grad():
        for old_l, new_l in zip(old_linears, new_linears):
            # old_l.weights tem shape (in_features, out_features); nn.Linear.weight
            # tem shape (out_features, in_features) -- daí a transposição.
            w = torch.tensor(old_l.weights.data, dtype=torch.float32).T
            b = torch.tensor(old_l.bias.data, dtype=torch.float32).reshape(-1)
            new_l.weight.copy_(w)
            new_l.bias.copy_(b)


def _build_matched_engines(arity: int = 1):
    set_backend("numpy")
    old_env = {
        name: Tensor(
            [[float((i + 1) * (j + 1)) / 10 for j in range(EMBEDDING_DIM)]],
            requires_grad=True,
        )
        for i, name in enumerate(ENTITIES)
    }
    old_predicate = OldSequential(
        OldLinear(arity * EMBEDDING_DIM, HIDDEN), OldReLU(), OldLinear(HIDDEN, 1), OldSigmoid()
    )
    old_interp = Interpreter({"P": old_predicate}, old_env)

    new_env = BatchedGroundingEnv(ENTITIES, EMBEDDING_DIM, seed=0)
    with torch.no_grad():
        for name in ENTITIES:
            new_env.table[new_env.vocab[name]] = torch.tensor(
                old_env[name].data[0], dtype=torch.float32
            )
    new_predicate = build_predicate_mlp(arity * EMBEDDING_DIM, HIDDEN, num_hidden_layers=1)
    _copy_linear_stack(old_predicate, new_predicate)
    new_interp = BatchedInterpreter({"P": new_predicate}, new_env)

    return old_interp, new_interp, new_env.vocab


def _old_scalar(interp: Interpreter, formula) -> float:
    result = interp.eval_formula(formula, {})
    return result._flatten(result.data)[0]


def _new_scalar(interp: BatchedInterpreter, formula, vocab) -> float:
    group = compile_formulas([formula], vocab)[0]
    return interp.eval_group(group).item()


class TestInterpreterBridge:
    def test_atom_arity_1_matches(self):
        old_interp, new_interp, vocab = _build_matched_engines(arity=1)
        formula = Atom("P", [Constant("a")])
        assert _new_scalar(new_interp, formula, vocab) == pytest.approx(
            _old_scalar(old_interp, formula), abs=1e-5
        )

    def test_atom_arity_2_concatenation_order_matches(self):
        old_interp, new_interp, vocab = _build_matched_engines(arity=2)
        # ordem importa: P(a,b) != P(b,a) se a concatenação estiver trocada
        f_ab = Atom("P", [Constant("a"), Constant("b")])
        f_ba = Atom("P", [Constant("b"), Constant("a")])
        for formula in (f_ab, f_ba):
            assert _new_scalar(new_interp, formula, vocab) == pytest.approx(
                _old_scalar(old_interp, formula), abs=1e-5
            )
        # e as duas ordens devem mesmo dar valores diferentes (senão o teste
        # acima não estaria de fato exercitando a ordem)
        assert _old_scalar(old_interp, f_ab) != pytest.approx(_old_scalar(old_interp, f_ba))

    def test_not_matches(self):
        old_interp, new_interp, vocab = _build_matched_engines(arity=1)
        formula = Not(Atom("P", [Constant("a")]))
        assert _new_scalar(new_interp, formula, vocab) == pytest.approx(
            _old_scalar(old_interp, formula), abs=1e-5
        )

    def test_implies_leaf_order_across_antecedent_and_consequent_matches(self):
        old_interp, new_interp, vocab = _build_matched_engines(arity=1)
        formula = Implies(Atom("P", [Constant("a")]), Atom("P", [Constant("b")]))
        assert _new_scalar(new_interp, formula, vocab) == pytest.approx(
            _old_scalar(old_interp, formula), abs=1e-5
        )
