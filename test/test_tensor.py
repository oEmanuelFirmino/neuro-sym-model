import sys
import logging
import pytest

try:
    from src.neurosym.tensor.tensor import Tensor
except ImportError:
    pytest.fail("❌ Erro ao importar Tensor", pytrace=False)


class TensorTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger("TensorBackpropTest")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

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
                lines.append(
                    f"{indent}    {formatted_item}{',' if i < len(data) - 1 else ''}"
                )
            lines.append(f"\n{indent}]")
            return "".join(lines)
        return f"{data:8.4f}" if isinstance(data, float) else str(data)

    def print_tensor_info(self, name: str, tensor: Tensor, description: str = ""):
        desc_text = f" ({description})" if description else ""
        self.logger.info(f"  🔹 Tensor: {name}{desc_text}")
        self.logger.info(
            f"     Shape: {tensor.shape}, Requires Grad: {tensor.requires_grad}"
        )
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


@pytest.fixture
def formatter():
    fmt = TensorTestFormatter()
    fmt.print_banner("Suite de Testes: Tensores & Autograd")
    return fmt


class TestTensorBackprop:
    def test_basic_operations(self, formatter):
        formatter.print_section_header("Operações Básicas com Backpropagation")

        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

        formatter.print_tensor_info("a", a, "Tensor A")
        formatter.print_tensor_info("b", b, "Tensor B")

        c = a + b
        loss = c.sum()

        formatter.print_operation_result("c = a + b", c)
        formatter.print_operation_result("loss = c.sum()", loss)

        expected_c = [[6.0, 8.0], [10.0, 12.0]]

        assert c._flatten(c.data) == pytest.approx(Tensor._flatten(expected_c))

        assert loss.data == pytest.approx(36.0)

        loss.backward()

        formatter.print_backward_info("a", a)
        formatter.print_backward_info("b", b)

        assert a.grad is not None
        flat_grad_a = a._flatten(a.grad.data)
        assert all(g == 1.0 for g in flat_grad_a)

    def test_multiplication(self, formatter):
        formatter.print_section_header("Multiplicação Matricial e Backpropagation")

        x = Tensor([[1.0, 2.0]], requires_grad=True)
        y = Tensor([[3.0], [4.0]], requires_grad=True)

        formatter.print_tensor_info("x", x, "Tensor X (1x2)")
        formatter.print_tensor_info("y", y, "Tensor Y (2x1)")

        z = x.dot(y)
        loss = z.sum()

        formatter.print_operation_result("z = x.dot(y)", z)

        assert z.data == [[11.0]]
        assert z.shape == (1, 1)

        loss.backward()

        formatter.print_backward_info("x", x)
        formatter.print_backward_info("y", y)

        assert x.grad._flatten(x.grad.data) == pytest.approx([3.0, 4.0])
        assert y.grad._flatten(y.grad.data) == pytest.approx([1.0, 2.0])

    def test_transpose(self, formatter):
        formatter.print_section_header("Transposição com Backpropagation")

        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        formatter.print_tensor_info("a", a, "Tensor A (2x3)")

        b = a.transpose()
        loss = b.sum()

        formatter.print_tensor_info("b = a.transpose()", b, "Tensor B (3x2)")

        assert b.shape == (3, 2)
        assert b.data[0][1] == 4.0

        loss.backward()
        formatter.print_backward_info("a", a)

        assert a.grad.shape == (2, 3)
        assert all(g == 1.0 for g in a._flatten(a.grad.data))

    def test_broadcasting(self, formatter):
        formatter.print_section_header("Broadcasting com Backpropagation")

        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([5.0, 6.0], requires_grad=True)

        formatter.print_tensor_info("x", x, "Matriz X (2x2)")
        formatter.print_tensor_info("b", b, "Vetor B (2,)")

        y = x + b
        loss = y.sum()

        formatter.print_operation_result("y = x + b", y)

        expected_y = [[6.0, 8.0], [8.0, 10.0]]

        assert y._flatten(y.data) == pytest.approx(Tensor._flatten(expected_y))

        loss.backward()

        formatter.print_backward_info("x", x)
        formatter.print_backward_info("b", b)

        assert b.grad._flatten(b.grad.data) == pytest.approx([2.0, 2.0])

    def test_complex_chain(self, formatter):
        formatter.print_section_header("Cadeia Complexa (Linear Layer)")

        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        w = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True)
        bias = Tensor([[0.7, 0.8]], requires_grad=True)

        formatter.print_tensor_info("x", x, "Input (1x3)")
        formatter.print_tensor_info("w", w, "Weights (3x2)")
        formatter.print_tensor_info("b", bias, "Bias (2,)")

        h = x.dot(w) + bias
        o = h.exp()
        loss = o.sum()

        formatter.print_operation_result("h = x.dot(w) + b", h)
        formatter.print_operation_result("o = h.exp()", o)
        formatter.print_operation_result("loss = o.sum()", loss)

        loss.backward()

        formatter.print_backward_info("x", x)
        formatter.print_backward_info("w", w)
        formatter.print_backward_info("b", bias)

        assert x.grad is not None
        assert w.grad is not None
        assert bias.grad is not None

        formatter.logger.info("  ✅ Gradientes complexos calculados com sucesso.")
