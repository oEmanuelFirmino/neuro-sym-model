"""Equivalente batched de `experiments/modular_addition/run.py`, pro motor
`torch_engine` -- mesma tarefa (Adição Modular Z_p, axiomas de comutatividade +
identidade, Product T-norm + L1), mas capaz de escalar hidden/embedding_dim e o
número de fatos sem cair no overhead de um `eval_formula` por fato do motor
original.

Reaproveita `dataset.py`/`axioms.py` sem nenhuma alteração -- essas duas funções
só constroem `Atom`/`Implies`/`Constant` (AST pura de `logic.py`), sem nenhuma
dependência do `Tensor` do motor antigo, então servem os dois motores igualmente.
Só `build_grounding_env` (que cria um `Tensor` por constante) e a métrica de
acurácia (que chama `Interpreter.eval_formula`) precisam de uma versão nova.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from experiments.modular_addition.axioms import commutativity_axioms, identity_axioms
from experiments.modular_addition.dataset import Fact, ModularAdditionData, generate_split
from src.neurosym.logic import Atom, Constant
from src.neurosym.torch_engine import (
    BatchedGroundingEnv,
    BatchedInterpreter,
    BatchedTrainer,
    DLGModel,
    build_predicate_mlp,
)
from src.neurosym.torch_engine.compile import compile_formulas


@dataclass
class TorchExperimentSpec:
    model: DLGModel
    trainer: BatchedTrainer
    rules: list
    facts: List[Fact]
    val_facts: List[Fact]
    test_facts: List[Fact]
    p: int


def make_batched_argmax_accuracy_fn(
    model: DLGModel, interpreter: BatchedInterpreter, p: int
) -> Callable[[List[Fact]], Optional[float]]:
    """Equivalente batched de `evaluation.make_argmax_accuracy_fn`: dado (a,b), o
    candidato c com maior grau de verdade dentre os p candidatos é o correto?
    Em vez de um `eval_formula` por candidato por fato (O(p) chamadas Python por
    fato), monta UM lote (N_positivos * p, 3) e avalia tudo de uma vez."""
    vocab = model.grounding_env.vocab

    def accuracy_fn(facts: List[Fact]) -> Optional[float]:
        positive_facts = [(formula, target) for formula, target in facts if target >= 0.5]
        if not positive_facts:
            return None

        candidates = []
        correct_c = []
        for formula, _ in positive_facts:
            a_name, b_name, c_true_name = (term.name for term in formula.terms)
            correct_c.append(int(c_true_name))
            for c in range(p):
                candidates.append(Atom("Add", [Constant(a_name), Constant(b_name), Constant(str(c))]))

        group = compile_formulas(candidates, vocab)[0]
        with torch.no_grad():
            scores = interpreter.eval_group(group).reshape(len(positive_facts), p)
        best_c = scores.argmax(dim=1)
        correct_c_t = torch.tensor(correct_c, dtype=torch.long, device=scores.device)
        return (best_c == correct_c_t).float().mean().item()

    return accuracy_fn


def _build_predicate(in_features: int, hidden: int) -> torch.nn.Sequential:
    return build_predicate_mlp(in_features, hidden, num_hidden_layers=2)


def build_torch_modular_addition_experiment(
    seed: int,
    p: int,
    embedding_dim: int,
    hidden: int,
    epochs: int,
    lr: float = 1e-3,
    train_frac: float = 0.5,
    val_frac: float = 0.25,
    negatives_per_positive: int = 1,
    use_axioms: bool = True,
    lambda_semantic: float = 1.0,
    gamma_l1: float = 1e-4,
    weight_decay: float = 1e-2,
    device: str = "cpu",
    val_eval_every: int = 1,
) -> TorchExperimentSpec:
    torch.manual_seed(seed)

    data: ModularAdditionData = generate_split(
        p, seed=seed, train_frac=train_frac, val_frac=val_frac,
        negatives_per_positive=negatives_per_positive,
    )

    entities = [str(i) for i in range(p)]
    grounding_env = BatchedGroundingEnv(entities, embedding_dim, seed=seed)
    predicate_map = {"Add": _build_predicate(3 * embedding_dim, hidden)}
    model = DLGModel(grounding_env, predicate_map)
    model.to(device)

    interpreter = BatchedInterpreter(predicate_map, grounding_env)
    accuracy_fn = make_batched_argmax_accuracy_fn(model, interpreter, p)

    rules = (
        commutativity_axioms(data.train_pairs, p) + identity_axioms(p) if use_axioms else []
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    trainer = BatchedTrainer(
        model, interpreter, optimizer, epochs=epochs, device=device,
        lambda_semantic=lambda_semantic, gamma_l1=gamma_l1,
        accuracy_fn=accuracy_fn, val_eval_every=val_eval_every,
    )

    return TorchExperimentSpec(
        model=model, trainer=trainer, rules=rules,
        facts=data.train_facts, val_facts=data.val_facts, test_facts=data.test_facts,
        p=p,
    )


def predict_add(model: DLGModel, interpreter: BatchedInterpreter, a: int, b: int, p: int) -> int:
    """Consulta de inferência: dado (a,b), retorna o c com maior grau de verdade
    previsto pelo predicado Add treinado -- é a mesma lógica de
    `make_batched_argmax_accuracy_fn`, exposta pra uma única consulta manual."""
    vocab = model.grounding_env.vocab
    candidates = [Atom("Add", [Constant(str(a)), Constant(str(b)), Constant(str(c))]) for c in range(p)]
    group = compile_formulas(candidates, vocab)[0]
    with torch.no_grad():
        scores = interpreter.eval_group(group)
    return int(scores.argmax().item())
