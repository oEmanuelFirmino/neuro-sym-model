import sys
import logging
import pytest

try:
    from src.neurosym.tensor.tensor import Tensor
    from src.neurosym.training.optimizer import SGD
except ImportError:
    pytest.fail(
        "❌ Erro ao importar um ou mais módulos necessários para o teste do otimizador.",
        pytrace=False,
    )


class OptimizerTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger("OptimizerTest")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def print_banner(self, title: str):
        self.logger.info("")
        self.logger.info("=" * 75)
        self.logger.info(f"  ⚙️  {title.upper()} ⚙️")
        self.logger.info("=" * 75)

    def print_section(self, title: str):
        self.logger.info(f"\n--- {title} ---")

    def print_tensor_info(self, name, tensor):
        flat_data = tensor._flatten(tensor.data)
        data_str = ", ".join([f"{x:.4f}" for x in flat_data])
        grad_str = "N/A"
        if tensor.grad:
            flat_grad = tensor.grad._flatten(tensor.grad.data)
            grad_str = ", ".join([f"{g:.4f}" for g in flat_grad])

        self.logger.info(f"  🔹 {name}:")
        self.logger.info(f"     Data: [{data_str}]")
        self.logger.info(f"     Grad: [{grad_str}]")


@pytest.fixture
def formatter():
    return OptimizerTestFormatter()


@pytest.fixture
def optimizer_setup(formatter):
    """
    Configura um estado inicial para os testes:
    - Parâmetros com valores iniciais
    - Gradientes pré-definidos (simulando um backward pass)
    - Otimizador SGD inicializado
    """
    formatter.print_section("Configuração Inicial (Fixture)")

    params = [
        Tensor([[0.5, -1.0]], requires_grad=True),
        Tensor(2.0, requires_grad=True),
    ]

    params[0].grad = Tensor([[0.1, 0.2]])
    params[1].grad = Tensor(-0.5)

    lr = 0.1
    optimizer = SGD(params, lr)

    formatter.print_tensor_info("Parâmetro 1 (inicial)", params[0])
    formatter.print_tensor_info("Parâmetro 2 (inicial)", params[1])
    formatter.logger.info(f"  🔹 Learning Rate (lr): {lr}")

    return optimizer, params, lr


class TestOptimizer:

    def test_sgd_step(self, optimizer_setup, formatter):
        formatter.print_banner("Teste do Otimizador SGD")
        formatter.print_section("Testando `optimizer.step()`")

        optimizer, params, lr = optimizer_setup

        expected_p1_data = [0.49, -1.02]

        expected_p2_data = [2.05]

        optimizer.step()

        formatter.print_tensor_info("Parâmetro 1 (depois)", params[0])
        formatter.print_tensor_info("Parâmetro 2 (depois)", params[1])

        flat_p1 = params[0]._flatten(params[0].data)
        flat_p2 = params[1]._flatten(params[1].data)

        assert flat_p1 == pytest.approx(
            expected_p1_data, abs=1e-6
        ), f"Falha na atualização P1. Esperado {expected_p1_data}, obtido {flat_p1}"

        assert flat_p2 == pytest.approx(
            expected_p2_data, abs=1e-6
        ), f"Falha na atualização P2. Esperado {expected_p2_data}, obtido {flat_p2}"

        formatter.logger.info("  ✅ Atualização de pesos verificada com sucesso.")

    def test_zero_grad(self, optimizer_setup, formatter):
        formatter.print_section("Testando `optimizer.zero_grad()`")

        optimizer, params, _ = optimizer_setup

        assert params[0].grad is not None
        flat_grad_before = params[0].grad._flatten(params[0].grad.data)
        assert any(
            g != 0 for g in flat_grad_before
        ), "Setup incorreto: gradientes deveriam ser não-nulos antes do teste."

        optimizer.zero_grad()

        formatter.print_tensor_info("Parâmetro 1 (após zero_grad)", params[0])
        formatter.print_tensor_info("Parâmetro 2 (após zero_grad)", params[1])

        for i, p in enumerate(params):
            assert p.grad is not None
            flat_grad = p.grad._flatten(p.grad.data)

            assert flat_grad == pytest.approx(
                [0.0] * len(flat_grad), abs=1e-9
            ), f"Gradiente do parâmetro {i+1} não foi zerado completamente: {flat_grad}"

        formatter.logger.info("  ✅ Zeragem de gradientes verificada com sucesso.")
