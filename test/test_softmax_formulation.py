import math
import sys
import logging
import pytest

try:
    from experiments.modular_addition.softmax_formulation import (
        build_classifier,
        make_softmax_build_fn,
        softmax_cross_entropy,
        softmax_probs,
    )
    from experiments.run_multiseed import run_single_seed
    from src.neurosym.tensor.tensor import Tensor
except ImportError:
    pytest.fail("❌ Erro ao importar a formulação softmax.", pytrace=False)


class SoftmaxTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger("SoftmaxFormulationTest")
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
    return SoftmaxTestFormatter()


class TestSoftmaxOps:
    def test_probs_sum_to_one(self, formatter):
        logits = Tensor([[1.0, 2.0, 3.0]])
        probs = softmax_probs(logits)
        flat = Tensor._flatten(probs.data)

        assert sum(flat) == pytest.approx(1.0)
        assert flat[2] > flat[1] > flat[0]
        formatter.logger.info("  ✅ softmax normaliza e preserva a ordem dos logits.")

    def test_cross_entropy_matches_manual(self, formatter):
        logits = Tensor([[1.0, 2.0, 3.0]], requires_grad=False)
        # CE(target=2) = log(sum exp(l)) - l_2
        expected = math.log(sum(math.exp(v) for v in [1.0, 2.0, 3.0])) - 3.0

        ce = softmax_cross_entropy(logits, target=2, p=3)
        assert ce.data == pytest.approx(expected, abs=1e-9)
        formatter.logger.info("  ✅ cross-entropy bate com o cálculo manual.")

    def test_cross_entropy_is_shift_invariant(self, formatter):
        a = softmax_cross_entropy(Tensor([[1.0, 2.0, 3.0]]), 1, 3)
        b = softmax_cross_entropy(Tensor([[101.0, 102.0, 103.0]]), 1, 3)
        assert a.data == pytest.approx(b.data, abs=1e-9)
        formatter.logger.info("  ✅ CE invariante a deslocamento constante (estável).")

    def test_cross_entropy_gradient_flows(self, formatter):
        embedding = Tensor([[0.5, -0.3]], requires_grad=True)
        classifier = build_classifier(embedding_dim=1, hidden=4, p=3)
        logits = classifier(embedding)

        ce = softmax_cross_entropy(logits, target=0, p=3)
        ce.backward()

        flat_grad = embedding.grad._flatten(embedding.grad.data)
        assert any(g != 0 for g in flat_grad)
        formatter.logger.info("  ✅ gradiente flui da CE até o embedding de entrada.")


class TestSoftmaxBuildFn:
    def test_end_to_end_tiny_run(self, formatter):
        build_fn = make_softmax_build_fn(
            p=5, use_axioms=True, embedding_dim=4, hidden=8, epochs=3, val_eval_every=1
        )
        result = run_single_seed(build_fn, seed=0)

        assert len(result["epoch_logs"]) == 3
        assert 0.0 <= result["final_val_accuracy"] <= 1.0
        assert 0.0 <= result["test_accuracy"] <= 1.0
        assert result["epoch_logs"][-1]["l_semantic"] > 0.0
        formatter.logger.info("  ✅ formulação softmax roda ponta a ponta no runner.")

    def test_no_axioms_variant_has_zero_semantic_loss(self, formatter):
        build_fn = make_softmax_build_fn(
            p=5, use_axioms=False, embedding_dim=4, hidden=8, epochs=2, val_eval_every=1
        )
        result = run_single_seed(build_fn, seed=0)

        assert result["epoch_logs"][-1]["l_semantic"] == pytest.approx(0.0)
        formatter.logger.info("  ✅ variante sem axiomas tem L_semantic == 0.")
