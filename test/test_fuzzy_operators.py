import sys
import logging
import pytest

try:
    from src.neurosym.tensor.tensor import Tensor
    from src.neurosym.interpreter.fuzzy_operators import (
        lukasiewicz_tnorm,
        lukasiewicz_tconorm,
        lukasiewicz_implication,
        get_operator,
        OPERATOR_MAP,
    )
except ImportError:
    pytest.fail("❌ Erro ao importar operadores fuzzy.", pytrace=False)


class FuzzyOperatorsTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger("FuzzyOperatorsTest")
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
    return FuzzyOperatorsTestFormatter()


class TestLukasiewiczOperators:
    def test_tnorm_values(self, formatter):
        # max(0, a + b - 1)
        assert lukasiewicz_tnorm(Tensor(0.7), Tensor(0.6)).data == pytest.approx(0.3)
        assert lukasiewicz_tnorm(Tensor(0.3), Tensor(0.3)).data == pytest.approx(0.0)
        formatter.logger.info("  ✅ lukasiewicz_tnorm bate com max(0, a+b-1).")

    def test_tconorm_values(self, formatter):
        # min(1, a + b)
        assert lukasiewicz_tconorm(Tensor(0.7), Tensor(0.6)).data == pytest.approx(1.0)
        assert lukasiewicz_tconorm(Tensor(0.2), Tensor(0.3)).data == pytest.approx(0.5)
        formatter.logger.info("  ✅ lukasiewicz_tconorm bate com min(1, a+b).")

    def test_implication_values(self, formatter):
        # min(1, 1 - a + b)
        assert lukasiewicz_implication(Tensor(0.3), Tensor(0.9)).data == pytest.approx(1.0)
        assert lukasiewicz_implication(Tensor(0.9), Tensor(0.3)).data == pytest.approx(
            0.4, abs=1e-6
        )
        formatter.logger.info("  ✅ lukasiewicz_implication bate com min(1, 1-a+b).")

    def test_gradient_flows_through_tnorm(self, formatter):
        a = Tensor(0.7, requires_grad=True)
        b = Tensor(0.6, requires_grad=True)
        out = lukasiewicz_tnorm(a, b)
        out.backward()

        # Na região a+b-1 > 0, d/da = d/db = 1 (regime linear do Lukasiewicz)
        assert a.grad.data == pytest.approx(1.0)
        assert b.grad.data == pytest.approx(1.0)
        formatter.logger.info("  ✅ gradiente não-nulo na região linear do Lukasiewicz.")

    def test_gradient_vanishes_in_flat_region(self, formatter):
        # a+b-1 < 0: regime plano, gradiente nulo -- exatamente a fragilidade que
        # o parecer do orientador (M4) contrasta com a Product T-norm.
        a = Tensor(0.1, requires_grad=True)
        b = Tensor(0.1, requires_grad=True)
        out = lukasiewicz_tnorm(a, b)
        out.backward()

        assert a.grad.data == pytest.approx(0.0)
        assert b.grad.data == pytest.approx(0.0)
        formatter.logger.info(
            "  ✅ gradiente nulo na região plana, confirmando o comportamento "
            "citado no artigo (Fig. 3) para contraste com a Product T-norm."
        )

    def test_registered_in_operator_map(self, formatter):
        assert get_operator("lukasiewicz_and") is lukasiewicz_tnorm
        assert get_operator("lukasiewicz_or") is lukasiewicz_tconorm
        assert get_operator("lukasiewicz_implies") is lukasiewicz_implication
        assert {"lukasiewicz_and", "lukasiewicz_or", "lukasiewicz_implies"} <= set(
            OPERATOR_MAP.keys()
        )
        formatter.logger.info("  ✅ operadores de Lukasiewicz registrados em OPERATOR_MAP.")
