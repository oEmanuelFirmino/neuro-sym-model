import sys
import logging
import pytest

try:
    from experiments.modular_addition.run import (
        make_dlg_build_fn,
        make_ltn_baseline_build_fn,
        make_mlp_baseline_build_fn,
    )
    from experiments.run_multiseed import ExperimentSpec, run_single_seed
    from src.neurosym.interpreter.fuzzy_operators import lukasiewicz_tnorm
except ImportError:
    pytest.fail(
        "❌ Erro ao importar os build_fns do experimento de Adição Modular.",
        pytrace=False,
    )


class ModularAdditionRunTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger("ModularAdditionRunTest")
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
    return ModularAdditionRunTestFormatter()


# p pequeno e poucas épocas para o teste rodar rápido -- não é um teste de
# qualidade do modelo, só de que a fiação (dataset -> Interpreter -> Trainer
# -> métricas) funciona ponta a ponta para cada arquitetura.
TINY_KWARGS = dict(p=5, embedding_dim=4, hidden=8, epochs=3, val_eval_every=1)


class TestBuildFns:
    def test_dlg_build_fn_wires_axioms_and_l1(self, formatter):
        build_fn = make_dlg_build_fn(**TINY_KWARGS, gamma_l1=0.1)
        spec = build_fn(0)

        assert isinstance(spec, ExperimentSpec)
        assert len(spec.rules) > 0  # comutatividade + identidade
        assert spec.trainer.gamma_l1 == pytest.approx(0.1)
        assert spec.trainer.interpreter.op_and(
            spec.trainer.interpreter.grounding_env["0"], spec.trainer.interpreter.grounding_env["0"]
        ) is not None  # product t-norm (default) responde normalmente

        formatter.logger.info("  ✅ build_fn do DLG usa axiomas + L1 > 0.")

    def test_mlp_baseline_has_no_rules_and_no_l1(self, formatter):
        build_fn = make_mlp_baseline_build_fn(**TINY_KWARGS)
        spec = build_fn(0)

        assert spec.rules == []
        assert spec.trainer.gamma_l1 == 0.0

        formatter.logger.info("  ✅ baseline MLP não usa regras nem L1.")

    def test_ltn_baseline_uses_lukasiewicz_operator(self, formatter):
        build_fn = make_ltn_baseline_build_fn(**TINY_KWARGS)
        spec = build_fn(0)

        assert spec.trainer.interpreter.op_and is lukasiewicz_tnorm
        assert spec.trainer.gamma_l1 == 0.0
        assert len(spec.rules) > 0  # ainda usa os axiomas, só troca o operador

        formatter.logger.info("  ✅ baseline LTN usa Lukasiewicz e não usa L1.")

    def test_all_three_run_single_seed_end_to_end(self, formatter):
        for name, factory in (
            ("dlg", make_dlg_build_fn),
            ("mlp_baseline", make_mlp_baseline_build_fn),
            ("ltn_baseline", make_ltn_baseline_build_fn),
        ):
            build_fn = factory(**TINY_KWARGS)
            result = run_single_seed(build_fn, seed=0)

            assert len(result["epoch_logs"]) == TINY_KWARGS["epochs"]
            assert 0.0 <= result["final_val_accuracy"] <= 1.0
            assert 0.0 <= result["test_accuracy"] <= 1.0
            formatter.logger.info(f"  ✅ {name} roda ponta a ponta sem erro.")
