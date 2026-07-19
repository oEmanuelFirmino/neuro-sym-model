import sys
import logging
import pytest

try:
    from experiments.modular_addition.fidelity import axiom_fidelity
    from src.neurosym.interpreter.interpreter import Interpreter
    from src.neurosym.module.module import Module
    from src.neurosym.tensor.tensor import Tensor
    from src.neurosym.training.metrics import weight_sparsity
except ImportError:
    pytest.fail("❌ Erro ao importar fidelidade/esparsidade.", pytrace=False)


class FidelitySparsityTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger("FidelitySparsityTest")
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
    return FidelitySparsityTestFormatter()


class TestWeightSparsity:
    def test_counts_near_zero_fraction(self, formatter):
        weights = [Tensor([[0.0, 0.0005, 0.5], [1.0, -0.0001, -2.0]])]
        # 3 de 6 pesos com |w| < 1e-3
        assert weight_sparsity(weights, eps=1e-3) == pytest.approx(0.5)
        formatter.logger.info("  ✅ fração de pesos quase nulos correta.")

    def test_empty_returns_none(self, formatter):
        assert weight_sparsity([], eps=1e-3) is None
        formatter.logger.info("  ✅ retorna None sem pesos.")


class _ExactAddPredicate(Module):
    """Predicado perfeito: conhece a soma modular; fidelidade deve ser 1.0."""

    def __init__(self, p: int):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        a, b, c = x.data[0]
        value = 1.0 if (int(round(a)) + int(round(b))) % self.p == int(round(c)) else 0.0
        return Tensor([[value]])


class TestAxiomFidelity:
    def test_perfect_predicate_has_full_fidelity(self, formatter):
        p = 7
        grounding_env = {str(i): Tensor([[float(i)]]) for i in range(p)}
        interpreter = Interpreter({"Add": _ExactAddPredicate(p)}, grounding_env)

        report = axiom_fidelity(interpreter, [(2, 3), (5, 6), (1, 4)], p)

        assert report["commutativity"]["mean"] == pytest.approx(1.0)
        assert report["identity"]["mean"] == pytest.approx(1.0)
        assert report["overall"]["mean"] == pytest.approx(1.0)
        formatter.logger.info("  ✅ predicado perfeito atinge fidelidade 1.0.")

    def test_implication_semantics(self, formatter):
        # Predicado que dá 0.8 para tudo: I(0.8, 0.8) = 1 - 0.8 + 0.64 = 0.84
        class Constant08(Module):
            def forward(self, x):
                return Tensor([[0.8]])

        p = 5
        grounding_env = {str(i): Tensor([[float(i)]]) for i in range(p)}
        interpreter = Interpreter({"Add": Constant08()}, grounding_env)

        report = axiom_fidelity(interpreter, [(1, 2)], p)
        assert report["commutativity"]["mean"] == pytest.approx(0.84)
        assert report["identity"]["mean"] == pytest.approx(0.8)
        formatter.logger.info("  ✅ semântica de Reichenbach aplicada na medição.")
