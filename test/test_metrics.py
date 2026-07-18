import sys
import logging
import pytest

try:
    from src.neurosym.training.metrics import (
        time_to_generalization,
        post_threshold_dip_count,
    )
except ImportError:
    pytest.fail("❌ Erro ao importar módulos de métricas.", pytrace=False)


class MetricsTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger("MetricsTest")
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
    return MetricsTestFormatter()


class TestTimeToGeneralization:
    def test_first_crossing_with_patience_one(self, formatter):
        curve = [0.1, 0.2, 0.96, 0.97, 0.98]
        t_g = time_to_generalization(curve, threshold=0.95, patience=1)
        assert t_g == 2
        formatter.logger.info("  ✅ patience=1 recupera o primeiro cruzamento simples.")

    def test_single_dip_delays_tg_with_patience(self, formatter):
        # Cruza o limiar no índice 2, mas cai de volta no índice 3 (flutuação
        # pós-transição). Com patience>=1 a janela [2:2+patience] falha porque
        # inclui o dip, então a detecção sustentada só ocorre depois do dip.
        curve = [0.1, 0.2, 0.96, 0.80, 0.96, 0.97, 0.98, 0.99]
        t_g_no_patience = time_to_generalization(curve, threshold=0.95, patience=1)
        assert t_g_no_patience == 2, "patience=1 é frágil e aceita o cruzamento isolado"

        t_g_with_patience = time_to_generalization(curve, threshold=0.95, patience=3)
        assert t_g_with_patience == 4, (
            "com patience=3, a primeira janela sustentada de 3 avaliações "
            "consecutivas >= threshold começa no índice 4"
        )
        formatter.logger.info(
            "  ✅ patience>1 ignora o cruzamento isolado e espera estabilidade."
        )

    def test_never_generalizes_returns_none(self, formatter):
        curve = [0.1, 0.2, 0.3, 0.4]
        assert time_to_generalization(curve, threshold=0.95, patience=2) is None
        formatter.logger.info("  ✅ retorna None quando o limiar nunca é sustentado.")

    def test_insufficient_trailing_window_returns_none(self, formatter):
        # O limiar é cruzado no último índice, mas não há avaliações suficientes
        # depois para formar uma janela completa de patience=3.
        curve = [0.1, 0.5, 0.96]
        assert time_to_generalization(curve, threshold=0.95, patience=3) is None
        formatter.logger.info(
            "  ✅ não aceita uma janela incompleta no final da curva como sustentada."
        )

    def test_invalid_patience_raises(self, formatter):
        with pytest.raises(ValueError):
            time_to_generalization([0.5], patience=0)
        formatter.logger.info("  ✅ patience < 1 levanta ValueError.")


class TestPostThresholdDipCount:
    def test_counts_dips_after_sustained_tg(self, formatter):
        curve = [0.1, 0.96, 0.97, 0.80, 0.99]
        dips = post_threshold_dip_count(curve, threshold=0.95, patience=2)
        # t_g sustentado (janela de 2) começa no índice 1 ([0.96, 0.97]);
        # a partir daí, apenas o índice 3 (0.80) fica abaixo do limiar.
        assert dips == 1
        formatter.logger.info("  ✅ conta corretamente as quedas após o T_g sustentado.")

    def test_returns_none_when_never_generalizes(self, formatter):
        curve = [0.1, 0.2]
        assert post_threshold_dip_count(curve, threshold=0.95, patience=1) is None
        formatter.logger.info("  ✅ retorna None quando não há T_g detectado.")
