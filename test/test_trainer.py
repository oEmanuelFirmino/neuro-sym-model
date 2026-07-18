import sys
import logging
import pytest

try:
    from src.neurosym.tensor.tensor import Tensor
    from src.neurosym.module.module import Module, Linear, Sigmoid
    from src.neurosym.logic.logic import Atom, Constant, Forall, Implies, Variable
    from src.neurosym.interpreter.interpreter import Interpreter
    from src.neurosym.training.optimizer import SGD
    from src.neurosym.training.trainer import Trainer
    from src.neurosym.training.callbacks import Callback
except ImportError:
    pytest.fail("❌ Erro ao importar módulos para o teste do Trainer.", pytrace=False)


class TrainerTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger("TrainerTest")
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
    return TrainerTestFormatter()


class PredicateNet(Module):
    def __init__(self, in_features):
        super().__init__()
        self.add_module("layer", Linear(in_features, 1))
        self.add_module("activation", Sigmoid())

    def forward(self, x):
        return self.activation(self.layer(x))


class LogCapture(Callback):
    def __init__(self):
        super().__init__()
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.history.append(dict(logs or {}))


def _build_env():
    embedding_dim = 2
    grounding_env = {
        "a": Tensor([[1.0, 0.0]], requires_grad=True),
        "b": Tensor([[0.0, 1.0]], requires_grad=True),
    }
    predicate_map = {"P": PredicateNet(embedding_dim)}
    return grounding_env, predicate_map


class TestTrainer:
    def test_loss_matches_l_total_formula(self, formatter):
        """loss == L_data + lambda*L_semantic + gamma*||W||_1 (artigo, Seção 4.4)."""
        grounding_env, predicate_map = _build_env()
        interpreter = Interpreter(predicate_map, grounding_env)

        facts = [(Atom("P", [Constant("a")]), 1.0)]
        x = Variable("x")
        rules = [Forall(x, Implies(Atom("P", [x]), Atom("P", [x])))]

        all_params = list(grounding_env.values()) + predicate_map["P"].parameters()
        capture = LogCapture()
        trainer = Trainer(
            interpreter,
            SGD(all_params, lr=0.0),
            epochs=1,
            lambda_semantic=2.0,
            gamma_l1=0.5,
            callbacks=[capture],
        )
        trainer.fit(rules=rules, facts=facts)

        logs = capture.history[-1]
        reconstructed = (
            logs["l_data"] + 2.0 * logs["l_semantic"] + 0.5 * logs["l1_penalty"]
        )
        assert logs["loss"] == pytest.approx(reconstructed, abs=1e-6)
        formatter.logger.info("  ✅ loss reconstruído a partir de L_data/L_semantic/L1 bate.")

    def test_gamma_l1_zero_ignores_penalty_in_loss_but_logs_it(self, formatter):
        grounding_env, predicate_map = _build_env()
        interpreter = Interpreter(predicate_map, grounding_env)
        facts = [(Atom("P", [Constant("a")]), 1.0)]
        all_params = list(grounding_env.values()) + predicate_map["P"].parameters()

        capture = LogCapture()
        trainer = Trainer(
            interpreter, SGD(all_params, lr=0.0), epochs=1, gamma_l1=0.0, callbacks=[capture]
        )
        trainer.fit(rules=[], facts=facts)

        logs = capture.history[-1]
        # l1_penalty is still measured (for monitoring) even though gamma_l1=0
        # keeps it out of the optimized loss.
        assert logs["l1_penalty"] > 0.0
        assert logs["loss"] == pytest.approx(logs["l_data"], abs=1e-6)
        formatter.logger.info(
            "  ✅ Com gamma_l1=0, a penalidade é registrada mas não afeta a loss."
        )

    def test_val_accuracy_tracked_per_epoch(self, formatter):
        grounding_env, predicate_map = _build_env()
        interpreter = Interpreter(predicate_map, grounding_env)
        facts = [(Atom("P", [Constant("a")]), 1.0)]
        val_facts = [
            (Atom("P", [Constant("a")]), 1.0),
            (Atom("P", [Constant("b")]), 0.0),
        ]
        all_params = list(grounding_env.values()) + predicate_map["P"].parameters()

        capture = LogCapture()
        trainer = Trainer(
            interpreter, SGD(all_params, lr=0.01), epochs=3, callbacks=[capture]
        )
        trainer.fit(rules=[], facts=facts, val_facts=val_facts)

        assert len(capture.history) == 3
        for epoch_logs in capture.history:
            assert "val_accuracy" in epoch_logs
            assert 0.0 <= epoch_logs["val_accuracy"] <= 1.0
        formatter.logger.info("  ✅ val_accuracy presente e no intervalo [0,1] em cada época.")

    def test_val_accuracy_absent_when_no_val_facts(self, formatter):
        grounding_env, predicate_map = _build_env()
        interpreter = Interpreter(predicate_map, grounding_env)
        facts = [(Atom("P", [Constant("a")]), 1.0)]
        all_params = list(grounding_env.values()) + predicate_map["P"].parameters()

        capture = LogCapture()
        trainer = Trainer(interpreter, SGD(all_params, lr=0.01), epochs=1, callbacks=[capture])
        trainer.fit(rules=[], facts=facts)

        assert "val_accuracy" not in capture.history[-1]
        formatter.logger.info("  ✅ val_accuracy ausente quando val_facts não é fornecido.")

    def test_evaluate_accuracy_binary_threshold(self, formatter):
        grounding_env, predicate_map = _build_env()
        interpreter = Interpreter(predicate_map, grounding_env)

        class FixedPredicate(Module):
            def __init__(self, value: float):
                super().__init__()
                self._value = value

            def forward(self, x):
                return Tensor([[self._value]])

        interpreter.predicate_map["Q"] = FixedPredicate(0.9)
        trainer = Trainer(interpreter, SGD([], lr=0.0), epochs=1)

        facts = [
            (Atom("Q", [Constant("a")]), 1.0),  # 0.9 >= 0.5 and 1.0 >= 0.5 -> correct
            (Atom("Q", [Constant("a")]), 0.0),  # 0.9 >= 0.5 but 0.0 < 0.5 -> incorrect
        ]
        accuracy = trainer.evaluate_accuracy(facts)
        assert accuracy == pytest.approx(0.5)
        formatter.logger.info("  ✅ evaluate_accuracy aplica o limiar binário corretamente.")
