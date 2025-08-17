import logging
import sys
from typing import Any, Union
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from .tensor import Tensor
except ImportError:
    try:
        from tensor import Tensor
    except ImportError:
        try:

            from src.tensor.tensor import Tensor
        except ImportError as e:
            print(f"❌ Erro ao importar Tensor: {e}")
            print("💡 Tentando execução direta...")
            print("💡 Execute: cd src/tensor && python test_tensor.py")
            sys.exit(1)


class TensorTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = self._setup_logger(log_level)

    def _setup_logger(self, log_level):
        logger = logging.getLogger("TensorAutogradTest")
        logger.setLevel(log_level)

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def print_banner(self, title: str):
        separator = "=" * 70
        self.logger.info("")
        self.logger.info(separator)
        self.logger.info(f"  🧮 {title.upper()}")
        self.logger.info(separator)

    def print_section_header(self, section_name: str):
        self.logger.info("")
        self.logger.info(f"📊 {section_name}")
        self.logger.info("-" * 55)

    def print_tensor_info(self, name: str, tensor, description: str = ""):
        desc_text = f" ({description})" if description else ""
        self.logger.info(f"🔧 {name}{desc_text}:")
        self.logger.info(f"   Shape: {tensor.shape}")
        self.logger.info(f"   Requires grad: {tensor.requires_grad}")
        if tensor._op:
            self.logger.info(f"   Operation: {tensor._op}")
        if tensor._parents:
            parent_info = [f"Tensor(shape={p.shape})" for p in tensor._parents]
            self.logger.info(f"   Parents: [{', '.join(parent_info)}]")
        self.logger.info(f"   Data: {self._format_tensor_data(tensor)}")

    def print_operation_result(
        self, operation: str, result: Any, description: str = ""
    ):
        desc_text = f" ({description})" if description else ""
        self.logger.info(f"")
        self.logger.info(f"🔹 {operation}{desc_text}:")

        if hasattr(result, "shape") and hasattr(result, "data"):

            self.logger.info(f"   Shape: {result.shape}")
            self.logger.info(f"   Requires grad: {result.requires_grad}")
            if result._op:
                self.logger.info(f"   Operation: {result._op}")
            self.logger.info(f"   Result: {self._format_tensor_data(result)}")
        else:

            if isinstance(result, float):
                self.logger.info(f"   Result: {result:.6f}")
            else:
                self.logger.info(f"   Result: {result}")

    def print_autograd_info(self, tensor, name: str = ""):
        name_text = f" for {name}" if name else ""
        self.logger.info(f"🔗 Autograd info{name_text}:")
        self.logger.info(f"   Requires grad: {tensor.requires_grad}")
        self.logger.info(f"   Has gradient: {tensor.grad is not None}")
        if tensor.grad is not None:
            self.logger.info(f"   Gradient: {tensor.grad}")
        self.logger.info(f"   Operation: {tensor._op or 'leaf node'}")
        self.logger.info(f"   Parents: {len(tensor._parents)} tensor(s)")

    def _format_tensor_data(self, tensor) -> str:
        return self._format_recursive(tensor.data, 0)

    def _format_recursive(self, data: Any, indent_level: int = 0) -> str:
        indent = "    " * indent_level

        if isinstance(data, list):
            if len(data) == 0:
                return "[]"

            if not isinstance(data[0], list):
                formatted_nums = [f"{x:8.4f}" for x in data]
                return "[" + ", ".join(formatted_nums) + "]"

            lines = [f"{indent}["]
            for i, item in enumerate(data):
                formatted_item = self._format_recursive(item, indent_level + 1)
                prefix = f"{indent}    "
                if i < len(data) - 1:
                    lines.append(f"{prefix}{formatted_item},")
                else:
                    lines.append(f"{prefix}{formatted_item}")
            lines.append(f"{indent}]")
            return "\n".join(lines)
        else:
            return f"{data:.4f}" if isinstance(data, float) else str(data)

    def print_separator(self):
        self.logger.info("─" * 55)

    def print_footer(self):
        separator = "=" * 70
        self.logger.info("")
        self.logger.info(separator)
        self.logger.info("✅ Todos os testes de autograd executados com sucesso!")
        self.logger.info("📝 Tensor autograd operations completed successfully")
        self.logger.info(separator)
        self.logger.info("")


class TensorAutogradTestSuite:
    def __init__(self, formatter: TensorTestFormatter):
        self.formatter = formatter

    def run_basic_operations_tests(self, t1, t2):
        self.formatter.print_section_header("Operações Básicas com Autograd")

        result_add = t1 + t2
        self.formatter.print_operation_result(
            "t1 + t2", result_add, "Soma elemento por elemento"
        )
        self.formatter.print_autograd_info(result_add, "soma")

        result_mul = t1 * 2
        self.formatter.print_operation_result(
            "t1 * 2", result_mul, "Multiplicação por escalar"
        )
        self.formatter.print_autograd_info(result_mul, "multiplicação")

        result_sub = t1 - t2
        self.formatter.print_operation_result(
            "t1 - t2", result_sub, "Subtração elemento por elemento"
        )

        result_radd = 3 + t1
        self.formatter.print_operation_result(
            "3 + t1", result_radd, "Soma reversa (escalar + tensor)"
        )

        result_rsub = 5 - t1
        self.formatter.print_operation_result(
            "5 - t1", result_rsub, "Subtração reversa (escalar - tensor)"
        )

    def run_matrix_operations_tests(self, t1, t2):
        self.formatter.print_section_header("Operações Matriciais")

        result_dot = t1.dot(t2)
        self.formatter.print_operation_result(
            "t1.dot(t2)", result_dot, "Produto matricial"
        )
        self.formatter.print_autograd_info(result_dot, "produto matricial")

    def run_mathematical_functions_tests(self, t1):
        self.formatter.print_section_header("Funções Matemáticas com Autograd")

        result_exp = t1.exp()
        self.formatter.print_operation_result(
            "t1.exp()", result_exp, "Função exponencial"
        )
        self.formatter.print_autograd_info(result_exp, "exponencial")

        result_log = t1.log()
        self.formatter.print_operation_result(
            "t1.log()", result_log, "Logaritmo natural"
        )
        self.formatter.print_autograd_info(result_log, "logaritmo")

        result_pow = t1.pow(2)
        self.formatter.print_operation_result(
            "t1.pow(2)", result_pow, "Elevar ao quadrado"
        )
        self.formatter.print_autograd_info(result_pow, "potência")

        result_pow_dec = t1.pow(0.5)
        self.formatter.print_operation_result(
            "t1.pow(0.5)", result_pow_dec, "Raiz quadrada"
        )

    def run_reduction_operations_tests(self, t1):
        self.formatter.print_section_header("Operações de Redução")

        result_sum = t1.sum()
        self.formatter.print_operation_result(
            "t1.sum()", result_sum, "Soma de todos os elementos"
        )

        result_mean = t1.mean()
        self.formatter.print_operation_result(
            "t1.mean()", result_mean, "Média de todos os elementos"
        )

    def run_autograd_specific_tests(self):
        self.formatter.print_section_header("Testes Específicos de Autograd")

        t_grad = Tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
        self.formatter.print_tensor_info(
            "t_grad", t_grad, "Tensor com requires_grad=True"
        )

        result_chain = t_grad * 2 + 1
        self.formatter.print_operation_result(
            "t_grad * 2 + 1", result_chain, "Cadeia de operações"
        )
        self.formatter.print_autograd_info(result_chain, "cadeia")

        t_grad.zero_grad()
        self.formatter.logger.info("🔄 Executado t_grad.zero_grad()")
        self.formatter.print_autograd_info(t_grad, "após zero_grad")

        detached = result_chain.detach()
        self.formatter.print_tensor_info(
            "detached", detached, "Tensor desconectado do grafo"
        )

        try:
            result_chain.backward()
        except NotImplementedError as e:
            self.formatter.logger.info(f"🚧 Backward ainda não implementado: {e}")

    def run_edge_cases_tests(self):
        self.formatter.print_section_header("Casos Extremos e Validações")

        t_scalar = Tensor(5.0, requires_grad=True)
        self.formatter.print_tensor_info(
            "t_scalar", t_scalar, "Tensor escalar com grad"
        )

        result_scalar_exp = t_scalar.exp()
        self.formatter.print_operation_result(
            "t_scalar.exp()", result_scalar_exp, "Exponencial do escalar"
        )

        t_1d = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        self.formatter.print_tensor_info("t_1d", t_1d, "Tensor 1D com grad")

        result_1d_sum = t_1d.sum()
        self.formatter.print_operation_result(
            "t_1d.sum()", result_1d_sum, "Soma do vetor 1D"
        )

        try:
            t_empty = Tensor([])
            self.formatter.print_tensor_info("t_empty", t_empty, "Tensor vazio")
        except Exception as e:
            self.formatter.logger.info(f"⚠️  Tensor vazio gerou erro: {e}")

        try:
            t_invalid = Tensor([[1, 2], [3, 4, 5]])
        except ValueError as e:
            self.formatter.logger.info(f"✅ Validação funcionou: {e}")

    def run_all_tests(self):
        self.formatter.print_banner(
            "Demonstração Completa da Classe Tensor com Autograd"
        )

        self.formatter.print_section_header("Inicialização dos Tensores")
        t1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        t2 = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

        self.formatter.print_tensor_info("t1", t1, "Tensor 2x2 - Matriz A com grad")
        self.formatter.print_tensor_info("t2", t2, "Tensor 2x2 - Matriz B com grad")

        self.run_basic_operations_tests(t1, t2)
        self.run_matrix_operations_tests(t1, t2)
        self.run_mathematical_functions_tests(t1)
        self.run_reduction_operations_tests(t1)
        self.run_autograd_specific_tests()
        self.run_edge_cases_tests()

        self.formatter.print_footer()


def main():
    try:

        formatter = TensorTestFormatter()
        test_suite = TensorAutogradTestSuite(formatter)

        test_suite.run_all_tests()

    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        print("💡 Certifique-se de que o arquivo 'tensor.py' está no mesmo diretório")
        print("💡 ou ajuste o caminho de importação no código")
    except Exception as e:
        print(f"❌ Erro durante a execução dos testes: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
