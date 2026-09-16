"""Espelha os testes de `test/test_trainer.py` para o `BatchedTrainer` -- mesma
composição de loss (L_total = L_data + lambda*L_semantic + gamma*||W||_1) e
mesmo contrato de logs/callbacks, agora sobre o motor batched."""

import pytest

torch = pytest.importorskip("torch")

from src.neurosym.logic import Atom, Constant
from src.neurosym.training.callbacks import Callback
from src.neurosym.torch_engine import (
    BatchedGroundingEnv,
    BatchedInterpreter,
    BatchedTrainer,
    DLGModel,
    build_predicate_mlp,
)


class LogCapture(Callback):
    def __init__(self):
        super().__init__()
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.history.append(dict(logs or {}))


def _build_experiment(lr: float = 0.0):
    entities = ["a", "b"]
    grounding_env = BatchedGroundingEnv(entities, embedding_dim=2, seed=0)
    predicate_map = {"P": build_predicate_mlp(2, hidden=4, num_hidden_layers=1)}
    model = DLGModel(grounding_env, predicate_map)
    interpreter = BatchedInterpreter(predicate_map, grounding_env)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, interpreter, optimizer


class TestBatchedTrainer:
    def test_loss_matches_l_total_formula(self):
        model, interpreter, optimizer = _build_experiment()
        facts = [(Atom("P", [Constant("a")]), 1.0)]
        rules = [Atom("P", [Constant("b")])]

        capture = LogCapture()
        trainer = BatchedTrainer(
            model,
            interpreter,
            optimizer,
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

    def test_gamma_l1_zero_ignores_penalty_in_loss_but_logs_it(self):
        model, interpreter, optimizer = _build_experiment()
        facts = [(Atom("P", [Constant("a")]), 1.0)]
        capture = LogCapture()
        trainer = BatchedTrainer(
            model, interpreter, optimizer, epochs=1, gamma_l1=0.0, callbacks=[capture]
        )
        trainer.fit(rules=[], facts=facts)

        logs = capture.history[-1]
        assert logs["l1_penalty"] > 0.0
        assert logs["loss"] == pytest.approx(logs["l_data"], abs=1e-6)

    def test_val_accuracy_tracked_per_epoch(self):
        model, interpreter, optimizer = _build_experiment(lr=0.01)
        facts = [(Atom("P", [Constant("a")]), 1.0)]
        val_facts = [
            (Atom("P", [Constant("a")]), 1.0),
            (Atom("P", [Constant("b")]), 0.0),
        ]
        capture = LogCapture()
        trainer = BatchedTrainer(model, interpreter, optimizer, epochs=3, callbacks=[capture])
        trainer.fit(rules=[], facts=facts, val_facts=val_facts)

        assert len(capture.history) == 3
        for epoch_logs in capture.history:
            assert "val_accuracy" in epoch_logs
            assert 0.0 <= epoch_logs["val_accuracy"] <= 1.0

    def test_val_accuracy_absent_when_no_val_facts(self):
        model, interpreter, optimizer = _build_experiment(lr=0.01)
        facts = [(Atom("P", [Constant("a")]), 1.0)]
        capture = LogCapture()
        trainer = BatchedTrainer(model, interpreter, optimizer, epochs=1, callbacks=[capture])
        trainer.fit(rules=[], facts=facts)

        assert "val_accuracy" not in capture.history[-1]

    def test_custom_accuracy_fn_overrides_default(self):
        model, interpreter, optimizer = _build_experiment()
        calls = []

        def custom_accuracy_fn(facts):
            calls.append(list(facts))
            return 0.42

        trainer = BatchedTrainer(
            model, interpreter, optimizer, epochs=1, accuracy_fn=custom_accuracy_fn
        )
        facts = [(Atom("P", [Constant("a")]), 1.0)]

        result = trainer.evaluate_accuracy(facts)

        assert result == pytest.approx(0.42)
        assert len(calls) == 1

    def test_val_eval_every_forward_fills_between_evaluations(self):
        model, interpreter, optimizer = _build_experiment()
        call_count = [0]

        def counting_accuracy_fn(facts):
            call_count[0] += 1
            return float(call_count[0])

        capture = LogCapture()
        trainer = BatchedTrainer(
            model,
            interpreter,
            optimizer,
            epochs=6,
            accuracy_fn=counting_accuracy_fn,
            val_eval_every=3,
            callbacks=[capture],
        )
        facts = [(Atom("P", [Constant("a")]), 1.0)]
        val_facts = [(Atom("P", [Constant("a")]), 1.0)]

        trainer.fit(rules=[], facts=facts, val_facts=val_facts)

        val_curve = [epoch["val_accuracy"] for epoch in capture.history]
        assert val_curve == [1.0, 1.0, 1.0, 2.0, 2.0, 3.0]
        assert call_count[0] == 3
