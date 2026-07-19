import sys
import logging
import pytest

try:
    from src.neurosym.tensor.tensor import Tensor
    from src.neurosym.module.module import Module, Linear, Sigmoid
    from src.neurosym.logic.logic import Atom, Constant
    from src.neurosym.interpreter.interpreter import Interpreter
    from src.neurosym.explainability.metrics import (
        compute_influences,
        concentration,
        deletion_curve,
        insertion_curve,
        curve_auc,
        evaluate_queries,
    )
except ImportError:
    pytest.fail("❌ Erro ao importar métricas de explicabilidade.", pytrace=False)


class ExplainabilityTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger("ExplainabilityMetricsTest")
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
    return ExplainabilityTestFormatter()


class SumPredicate(Module):
    """Predicado determinístico: sigmoid(soma das entradas). O gradiente flui
    apenas para as constantes que aparecem na consulta."""

    def forward(self, x: Tensor) -> Tensor:
        return Sigmoid()(x.sum())


def _build_env():
    grounding_env = {
        "a": Tensor([[0.5, 0.5]], requires_grad=True),
        "b": Tensor([[0.3, -0.2]], requires_grad=True),
        "c": Tensor([[-0.4, 0.1]], requires_grad=True),
    }
    predicate_map = {"P": SumPredicate()}
    return Interpreter(predicate_map, grounding_env)


class TestComputeInfluences:
    def test_gradient_only_on_query_constants(self, formatter):
        interpreter = _build_env()
        query = Atom("P", [Constant("a")])

        influences = compute_influences(interpreter, query)

        assert influences["a"] > 0.0
        assert influences["b"] == pytest.approx(0.0)
        assert influences["c"] == pytest.approx(0.0)
        formatter.logger.info("  ✅ gradiente concentrado apenas na constante consultada.")


class TestConcentration:
    def test_full_concentration_when_only_relevant_has_mass(self, formatter):
        influences = {"a": 2.0, "b": 0.0, "c": 0.0}
        assert concentration(influences, {"a"}) == pytest.approx(1.0)
        formatter.logger.info("  ✅ concentração 1.0 quando toda a massa é relevante.")

    def test_partial_concentration(self, formatter):
        influences = {"a": 3.0, "b": 1.0}
        assert concentration(influences, {"a"}) == pytest.approx(0.75)
        formatter.logger.info("  ✅ concentração parcial calculada corretamente.")

    def test_none_when_no_mass(self, formatter):
        assert concentration({"a": 0.0}, {"a"}) is None
        formatter.logger.info("  ✅ retorna None sem massa de gradiente.")


class TestDeletionInsertion:
    def test_deletion_restores_embeddings(self, formatter):
        interpreter = _build_env()
        original = {
            name: [list(row) for row in t.data]
            for name, t in interpreter.grounding_env.items()
        }
        query = Atom("P", [Constant("a")])

        curve = deletion_curve(interpreter, query, ["a", "b", "c"])

        assert len(curve) == 4  # baseline + 3 deleções
        for name, data in original.items():
            assert interpreter.grounding_env[name].data == data
        formatter.logger.info("  ✅ deletion_curve restaura os embeddings ao final.")

    def test_deleting_relevant_changes_truth(self, formatter):
        interpreter = _build_env()
        query = Atom("P", [Constant("a")])

        curve_relevant = deletion_curve(interpreter, query, ["a"])
        curve_irrelevant = deletion_curve(interpreter, query, ["b"])

        # deletar 'a' muda o valor da consulta P(a); deletar 'b' não.
        assert curve_relevant[1] != pytest.approx(curve_relevant[0])
        assert curve_irrelevant[1] == pytest.approx(curve_irrelevant[0])
        formatter.logger.info(
            "  ✅ deleção da constante relevante altera a predição; irrelevante não."
        )

    def test_insertion_is_reverse_of_deletion_extremes(self, formatter):
        interpreter = _build_env()
        query = Atom("P", [Constant("a")])

        deletion = deletion_curve(interpreter, query, ["a"])
        insertion = insertion_curve(interpreter, query, ["a"])

        # o fim da insertion (tudo restaurado) == início da deletion (nada deletado)
        assert insertion[-1] == pytest.approx(deletion[0])
        # o início da insertion (tudo zerado) == fim da deletion (tudo deletado)
        assert insertion[0] == pytest.approx(deletion[-1])
        formatter.logger.info("  ✅ extremos de insertion/deletion são consistentes.")

    def test_curve_auc(self, formatter):
        assert curve_auc([1.0, 0.5, 0.0]) == pytest.approx(0.5)
        assert curve_auc([]) == 0.0
        formatter.logger.info("  ✅ AUC normalizada correta.")


class TestEvaluateQueries:
    def test_aggregates_over_queries(self, formatter):
        interpreter = _build_env()
        queries = [
            (Atom("P", [Constant("a")]), {"a"}),
            (Atom("P", [Constant("b")]), {"b"}),
        ]

        report = evaluate_queries(interpreter, queries, seed=0)

        assert report["n_queries"] == 2
        # gradiente flui só para a constante consultada -> concentração perfeita
        assert report["concentration"]["mean"] == pytest.approx(1.0)
        assert report["concentration"]["n"] == 2
        for key in (
            "deletion_auc_gradient",
            "deletion_auc_random",
            "insertion_auc_gradient",
            "insertion_auc_random",
        ):
            assert report[key]["mean"] is not None
        assert len(report["per_query"]) == 2
        formatter.logger.info(
            "  ✅ evaluate_queries agrega concentração e AUCs sobre N consultas."
        )
