import sys
import logging
import pytest

try:
    from experiments.kinship.domain import (
        DEFAULT_EDGES,
        ancestor_proof_formula,
        build_kinship_grounding_env,
        generate_kinship,
    )
    from src.neurosym.explainability.metrics import compute_influences
    from src.neurosym.interpreter.interpreter import Interpreter
    from src.neurosym.logic.logic import Atom, Constant, Or
    from src.neurosym.module.module import Module, Linear, Sigmoid
    from src.neurosym.tensor.tensor import Tensor
except ImportError:
    pytest.fail("❌ Erro ao importar o domínio de parentesco.", pytrace=False)


class KinshipTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger("KinshipDomainTest")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)


@pytest.fixture
def formatter():
    return KinshipTestFormatter()


class PredicateNet(Module):
    def __init__(self, in_features):
        super().__init__()
        self.add_module("layer", Linear(in_features, 1))
        self.add_module("activation", Sigmoid())

    def forward(self, x):
        return self.activation(self.layer(x))


def _build_interpreter(embedding_dim=4, seed=0):
    data = generate_kinship(seed=seed)
    grounding_env = build_kinship_grounding_env(data.entities, embedding_dim, seed)
    predicate_map = {
        "parent": PredicateNet(2 * embedding_dim),
        "ancestor_flat": PredicateNet(2 * embedding_dim),
    }
    return data, Interpreter(predicate_map, grounding_env)


class TestGenerateKinship:
    def test_facts_and_queries_structure(self, formatter):
        data = generate_kinship(seed=0)

        n_edges = len(DEFAULT_EDGES)
        positives = [f for f, t in data.parent_facts if t == 1.0]
        negatives = [f for f, t in data.parent_facts if t == 0.0]
        assert len(positives) == n_edges
        assert len(negatives) == n_edges * 2  # negatives_per_positive=2

        # a0->a1->a2->a3: pares encadeados devem incluir (a0,a2), (a0,a3), (a1,a3)
        chained = {(x, z) for x, z, _ in data.chained_queries}
        assert ("a0", "a2") in chained
        assert ("a0", "a3") in chained
        assert ("a1", "a3") in chained
        # pares diretos (arestas) não são consultas encadeadas
        assert ("a0", "a1") not in chained
        formatter.logger.info("  ✅ fatos e consultas encadeadas gerados corretamente.")

    def test_intermediates_are_correct(self, formatter):
        data = generate_kinship(seed=0)
        by_pair = {(x, z): mids for x, z, mids in data.chained_queries}

        assert by_pair[("a0", "a2")] == {"a1"}
        assert by_pair[("a0", "a3")] == {"a1", "a2"}
        assert by_pair[("b1", "b3")] == {"b2"}
        formatter.logger.info("  ✅ intermediários dos caminhos identificados corretamente.")

    def test_deterministic_by_seed(self, formatter):
        data_a = generate_kinship(seed=5)
        data_b = generate_kinship(seed=5)
        assert [repr(f) for f, _ in data_a.parent_facts] == [
            repr(f) for f, _ in data_b.parent_facts
        ]
        formatter.logger.info("  ✅ geração determinística por seed.")


class TestAncestorProofFormula:
    def test_depth_one_is_plain_atom(self, formatter):
        formula = ancestor_proof_formula("a0", "a1", ["a0", "a1", "a2"], depth=1)
        assert isinstance(formula, Atom)
        assert formula.predicate_name == "parent"
        formatter.logger.info("  ✅ profundidade 1 reduz a parent(x,z).")

    def test_depth_two_is_disjunction_over_intermediates(self, formatter):
        entities = ["a0", "a1", "a2", "b0"]
        formula = ancestor_proof_formula("a0", "a2", entities, depth=2)
        assert isinstance(formula, Or)
        # texto da fórmula deve mencionar os intermediários possíveis (a1, b0)
        text = repr(formula)
        assert "a1" in text and "b0" in text
        formatter.logger.info("  ✅ profundidade 2 compõe disjunção sobre intermediários.")

    def test_gradient_reaches_intermediates_composed_but_not_flat(self, formatter):
        """A propriedade central do domínio: na consulta composta o gradiente
        alcança o intermediário do caminho; no predicado plano, não."""
        data, interpreter = _build_interpreter()

        composed = ancestor_proof_formula("a0", "a2", data.entities, depth=2)
        influences_composed = compute_influences(interpreter, composed)

        flat = Atom("ancestor_flat", [Constant("a0"), Constant("a2")])
        influences_flat = compute_influences(interpreter, flat)

        # composta: massa em a0, a2 E no intermediário a1
        assert influences_composed["a0"] > 0.0
        assert influences_composed["a2"] > 0.0
        assert influences_composed["a1"] > 0.0

        # plana: massa só em a0 e a2; a1 estruturalmente zero
        assert influences_flat["a0"] > 0.0
        assert influences_flat["a2"] > 0.0
        assert influences_flat["a1"] == pytest.approx(0.0)

        formatter.logger.info(
            "  ✅ gradiente alcança o intermediário na inferência composta "
            "(DAG de prova) e não no predicado plano — evidência arquitetural."
        )
