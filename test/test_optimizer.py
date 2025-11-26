import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.tensor.tensor import Tensor
    from src.training.optimizer import SGD
except ImportError:
    print(
        "❌ Erro ao importar um ou mais módulos necessários para o teste do otimizador."
    )
    sys.exit(1)


class OptimizerTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = self._setup_logger(log_level)

    def _setup_logger(self, log_level):
        logger = logging.getLogger("OptimizerTest")
        logger.setLevel(log_level)
        if logger.hasHandlers():
            logger.handlers.clear()
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

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

    def print_result(self, test_name, success, message=""):
        status = "✅ SUCESSO" if success else "❌ FALHA"
        self.logger.info(f"  ▶️  Teste '{test_name}': {status}")
        if message:
            self.logger.info(f"     {message}")


class OptimizerTestSuite:
    def __init__(self):
        self.formatter = OptimizerTestFormatter()

    def _are_lists_close(self, list1, list2, tol=1e-6):
        if len(list1) != len(list2):
            return False
        return all(abs(a - b) < tol for a, b in zip(list1, list2))

    def run_tests(self):
        self.formatter.print_banner("Teste do Otimizador SGD")

        self.formatter.print_section("1. Configuração Inicial")
        params = [
            Tensor([[0.5, -1.0]], requires_grad=True),
            Tensor(2.0, requires_grad=True),
        ]
        params[0].grad = Tensor([[0.1, 0.2]])
        params[1].grad = Tensor(-0.5)

        lr = 0.1
        optimizer = SGD(params, lr)

        self.formatter.print_tensor_info("Parâmetro 1 (antes)", params[0])
        self.formatter.print_tensor_info("Parâmetro 2 (antes)", params[1])
        self.formatter.logger.info(f"  🔹 Learning Rate (lr): {lr}")

        self.formatter.print_section("2. Testando `optimizer.step()`")

        expected_p1_data = [0.5 - lr * 0.1, -1.0 - lr * 0.2]
        expected_p2_data = [2.0 - lr * -0.5]

        optimizer.step()

        self.formatter.print_tensor_info("Parâmetro 1 (depois)", params[0])
        self.formatter.print_tensor_info("Parâmetro 2 (depois)", params[1])

        p1_updated = self._are_lists_close(
            params[0]._flatten(params[0].data), expected_p1_data
        )
        p2_updated = self._are_lists_close([params[1].data], expected_p2_data)

        self.formatter.print_result("Atualização de pesos", p1_updated and p2_updated)

        self.formatter.print_section("3. Testando `optimizer.zero_grad()`")
        optimizer.zero_grad()

        self.formatter.print_tensor_info("Parâmetro 1 (após zero_grad)", params[0])
        self.formatter.print_tensor_info("Parâmetro 2 (após zero_grad)", params[1])

        p1_grad_zero = self._are_lists_close(
            params[0].grad._flatten(params[0].grad.data), [0.0, 0.0]
        )
        p2_grad_zero = self._are_lists_close([params[1].grad.data], [0.0])

        self.formatter.print_result(
            "Zeragem de gradientes", p1_grad_zero and p2_grad_zero
        )

        self.formatter.logger.info("\n🎉 Testes do otimizador concluídos!")


def main():
    try:
        suite = OptimizerTestSuite()
        suite.run_tests()
    except Exception as e:
        print(f"\n❌ Erro fatal durante os testes do otimizador: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
