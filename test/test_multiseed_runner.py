import json
import random
import sys
import logging
import pytest

try:
    from src.neurosym.tensor.tensor import Tensor
    from src.neurosym.module.module import Module, Linear, Sigmoid
    from src.neurosym.logic.logic import Atom, Constant
    from src.neurosym.interpreter.interpreter import Interpreter
    from src.neurosym.training.optimizer import SGD
    from src.neurosym.training.trainer import Trainer
    from experiments.run_multiseed import ExperimentSpec, run_multiseed, run_single_seed
except ImportError:
    pytest.fail("❌ Erro ao importar módulos para o teste do runner multi-seed.", pytrace=False)


class RunnerTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger("MultiseedRunnerTest")
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
    return RunnerTestFormatter()


class PredicateNet(Module):
    def __init__(self, in_features):
        super().__init__()
        self.add_module("layer", Linear(in_features, 1))
        self.add_module("activation", Sigmoid())

    def forward(self, x):
        return self.activation(self.layer(x))


def _toy_build_fn(seed: int) -> ExperimentSpec:
    """Experimento de brinquedo, agnóstico de domínio, só para exercitar o
    runner sem depender do exemplo Socrates (que hoje não tem config.yaml)."""
    embedding_dim = 2
    grounding_env = {
        "a": Tensor([[random.uniform(-1, 1), random.uniform(-1, 1)]], requires_grad=True),
        "b": Tensor([[random.uniform(-1, 1), random.uniform(-1, 1)]], requires_grad=True),
    }
    predicate_map = {"P": PredicateNet(embedding_dim)}
    interpreter = Interpreter(predicate_map, grounding_env)

    facts = [(Atom("P", [Constant("a")]), 1.0)]
    val_facts = [
        (Atom("P", [Constant("a")]), 1.0),
        (Atom("P", [Constant("b")]), 0.0),
    ]
    test_facts = [(Atom("P", [Constant("b")]), 0.0)]

    all_params = list(grounding_env.values()) + predicate_map["P"].parameters()
    trainer = Trainer(
        interpreter, SGD(all_params, lr=0.05), epochs=5, gamma_l1=0.1
    )

    return ExperimentSpec(
        interpreter=interpreter,
        trainer=trainer,
        rules=[],
        facts=facts,
        val_facts=val_facts,
        test_facts=test_facts,
    )


class TestRunSingleSeed:
    def test_returns_expected_keys(self, formatter):
        result = run_single_seed(_toy_build_fn, seed=0)

        for key in (
            "seed",
            "t_g",
            "final_val_accuracy",
            "test_accuracy",
            "final_l1_penalty",
            "epoch_logs",
        ):
            assert key in result

        assert result["seed"] == 0
        assert len(result["epoch_logs"]) == 5
        assert 0.0 <= result["final_val_accuracy"] <= 1.0
        assert result["final_l1_penalty"] >= 0.0
        formatter.logger.info("  ✅ run_single_seed devolve todas as chaves esperadas.")

    def test_same_seed_is_deterministic(self, formatter):
        result_a = run_single_seed(_toy_build_fn, seed=42)
        result_b = run_single_seed(_toy_build_fn, seed=42)

        assert result_a["final_val_accuracy"] == pytest.approx(
            result_b["final_val_accuracy"]
        )
        assert result_a["epoch_logs"][-1]["loss"] == pytest.approx(
            result_b["epoch_logs"][-1]["loss"]
        )
        formatter.logger.info("  ✅ a mesma seed reproduz o mesmo resultado.")


class TestRunMultiseed:
    def test_aggregates_across_seeds(self, formatter):
        result = run_multiseed(_toy_build_fn, seeds=[0, 1, 2])

        assert result["aggregate"]["n_seeds"] == 3
        assert len(result["runs"]) == 3
        assert result["aggregate"]["seeds"] == [0, 1, 2]

        for metric in ("final_val_accuracy", "final_l1_penalty", "test_accuracy"):
            stats = result["aggregate"][metric]
            assert stats["n"] == 3
            assert stats["mean"] is not None
            assert stats["std"] is not None

        formatter.logger.info("  ✅ run_multiseed agrega média/desvio padrão sobre 3 seeds.")

    def test_serializes_to_json(self, formatter, tmp_path):
        output_path = tmp_path / "results.json"
        run_multiseed(_toy_build_fn, seeds=[0, 1], output_path=str(output_path))

        assert output_path.exists()
        with open(output_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["aggregate"]["n_seeds"] == 2
        assert len(loaded["runs"]) == 2
        formatter.logger.info("  ✅ resultados agregados serializados corretamente em JSON.")
