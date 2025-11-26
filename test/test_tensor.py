import logging
import sys

try:
    from src.neurosym.tensor.tensor import Tensor
except ImportError:
    print(f"❌ Erro ao importar Tensor")
    sys.exit(1)


class TensorTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = self._setup_logger(log_level)

    def _setup_logger(self, log_level):
        logger = logging.getLogger("TensorBackpropTest")
        logger.setLevel(log_level)
        if logger.hasHandlers():
            logger.handlers.clear()
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def print_banner(self, title: str):
        separator = "=" * 75
        self.logger.info("")
        self.logger.info(separator)
        self.logger.info(f"  🧠 {title.upper()} 🧠")
        self.logger.info(separator)

    def print_section_header(self, section_name: str):
        self.logger.info("")
        self.logger.info(f"▶️  {section_name}")
        self.logger.info("<" + "-" * 60)

    def _format_data(self, data, indent_level=0):
        if isinstance(data, list):
            if not data or not isinstance(data[0], list):
                return "[" + ", ".join([f"{x:8.4f}" for x in data]) + "]"
            indent = "    " * (indent_level + 1)
            lines = [f"\n{indent}["]
            for i, item in enumerate(data):
                formatted_item = self._format_data(item, indent_level + 1).strip()
                lines.append(f"{indent}    {formatted_item}{',' if i < len(data) - 1 else ''}")
            lines.append(f"\n{indent}]")
            return "".join(lines)
        return f"{data:8.4f}" if isinstance(data, float) else str(data)

    def print_tensor_info(self, name: str, tensor: Tensor, description: str = ""):
        desc_text = f" ({description})" if description else ""
        self.logger.info(f"  🔹 Tensor: {name}{desc_text}")
        self.logger.info(f"     Shape: {tensor.shape}, Requires Grad: {tensor.requires_grad}")
        self.logger.info(f"     Data: {self._format_data(tensor.data)}")

    def print_operation_result(self, operation: str, result: Tensor):
        self.logger.info(f"  🔹 Operação: {operation}")
        self.logger.info(f"     Resultado: {self._format_data(result.data)}")

    def print_backward_info(self, name: str, tensor: Tensor):
        self.logger.info(f"  🔹 Gradiente Final para '{name}':")
        if tensor.grad:
            self.logger.info(f"     ∇{name}: {self._format_data(tensor.grad.data)}")
        else:
            self.logger.info(f"     ∇{name}: None")

    def print_footer(self):
        separator = "=" * 75
        self.logger.info("")
        self.logger.info(separator)
        self.logger.info("  ✅ Todos os testes de backpropagation executados com sucesso!")
        self.logger.info(separator)
        self.logger.info("")


class TensorBackpropTestSuite:
    def __init__(self):
        self.formatter = TensorTestFormatter()

    def run_basic_operations_tests(self):
        self.formatter.print_section_header("Operações Básicas com Backpropagation")
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        self.formatter.print_tensor_info("a", a, "Tensor A")
        self.formatter.print_tensor_info("b", b, "Tensor B")

        c = a + b
        loss = c.sum()

        self.formatter.print_operation_result("c = a + b", c)
        self.formatter.print_operation_result("loss = c.sum()", loss)

        loss.backward()

        self.formatter.print_backward_info("a", a)
        self.formatter.print_backward_info("b", b)

    def run_multiplication_tests(self):
        self.formatter.print_section_header("Multiplicação Matricial e Backpropagation")
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        y = Tensor([[3.0], [4.0]], requires_grad=True)
        self.formatter.print_tensor_info("x", x, "Tensor X (1x2)")
        self.formatter.print_tensor_info("y", y, "Tensor Y (2x1)")

        z = x.dot(y)
        loss = z.sum()

        self.formatter.print_operation_result("z = x.dot(y)", z)
        self.formatter.print_operation_result("loss = z.sum()", loss)

        loss.backward()

        self.formatter.print_backward_info("x", x)
        self.formatter.print_backward_info("y", y)

    def run_transpose_tests(self):
        self.formatter.print_section_header("Transposição com Backpropagation")
        a = Tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
        self.formatter.print_tensor_info("a", a, "Tensor A (2x3)")

        b = a.transpose()
        loss = b.sum()
        self.formatter.print_tensor_info("b = a.transpose()", b, "Tensor B (3x2)")
        self.formatter.print_operation_result("loss = b.sum()", loss)

        loss.backward()

        self.formatter.print_backward_info("a", a)

    def run_broadcasting_tests(self):
        self.formatter.print_section_header("Broadcasting com Backpropagation")
        x = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
        b = Tensor([5., 6.], requires_grad=True)
        self.formatter.print_tensor_info("x", x, "Matriz X (2x2)")
        self.formatter.print_tensor_info("b", b, "Vetor B (2,)")

        y = x + b
        loss = y.sum()
        self.formatter.print_operation_result("y = x + b", y)
        self.formatter.print_operation_result("loss = y.sum()", loss)

        loss.backward()

        self.formatter.print_backward_info("x", x)
        self.formatter.print_backward_info("b", b)

    def run_complex_chain_tests(self):
        self.formatter.print_section_header("Cadeia Complexa (Linear Layer)")
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        w = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True)
        b = Tensor([[0.7, 0.8]], requires_grad=True)
        self.formatter.print_tensor_info("x", x, "Input (1x3)")
        self.formatter.print_tensor_info("w", w, "Weights (3x2)")
        self.formatter.print_tensor_info("b", b, "Bias (2,)")

        h = x.dot(w) + b
        o = h.exp()
        loss = o.sum()

        self.formatter.print_operation_result("h = x.dot(w) + b", h)
        self.formatter.print_operation_result("o = h.exp()", o)
        self.formatter.print_operation_result("loss = o.sum()", loss)

        loss.backward()

        self.formatter.print_backward_info("x", x)
        self.formatter.print_backward_info("w", w)
        self.formatter.print_backward_info("b", b)

    def run_all_tests(self):
        self.formatter.print_banner("Demonstração Completa de Backpropagation")
        self.run_basic_operations_tests()
        self.run_multiplication_tests()
        self.run_transpose_tests()
        self.run_broadcasting_tests()
        self.run_complex_chain_tests()
        self.formatter.print_footer()


def main():
    try:
        test_suite = TensorBackpropTestSuite()
        test_suite.run_all_tests()
    except Exception as e:
        print(f"❌ Erro fatal durante a execução dos testes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()