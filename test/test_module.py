import sys
import logging
import pytest

try:
    from src.neurosym.tensor.tensor import Tensor
    from src.neurosym.module.module import Module, Linear, Sigmoid, ReLU
except ImportError:
    pytest.fail("❌ Erro ao importar módulos.", pytrace=False)


class ModuleTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger("ModuleTest")
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
        self.logger.info(f"  🧠 {title.upper()} 🧠")
        self.logger.info("=" * 75)

    def print_section_header(self, section_name: str):
        self.logger.info("")
        self.logger.info(f"▶️  {section_name}")
        self.logger.info("<" + "-" * 60)

    def _format_data(self, data):
        if isinstance(data, list):
            if not data or not isinstance(data[0], list):
                return f"[{', '.join([f'{x: .4f}' for x in data])}]"
            return (
                "[\n"
                + ",\n".join([f"  {self._format_data(row)}" for row in data])
                + "\n]"
            )
        return f"{data: .4f}" if isinstance(data, float) else str(data)

    def print_tensor_info(self, name: str, tensor: Tensor):
        self.logger.info(
            f"  🔹 Tensor: {name} (Shape: {tensor.shape}, Grad: {tensor.requires_grad})"
        )
        self.logger.info(f"     Data: {self._format_data(tensor.data)}")
        if tensor.grad:
            self.logger.info(f"     Grad: {self._format_data(tensor.grad.data)}")


@pytest.fixture
def formatter():
    return ModuleTestFormatter()


class TestModule:
    def test_linear_layer(self, formatter):
        formatter.print_section_header("Teste da Camada Linear (Linear Layer)")

        in_features, out_features = 3, 2
        linear_layer = Linear(in_features, out_features)

        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        formatter.print_tensor_info("Input (x)", x)

        formatter.logger.info("  ⚙️  Executando forward pass...")
        output = linear_layer(x)
        formatter.print_tensor_info("Output", output)

        assert output.shape == (1, out_features)
        assert output.requires_grad

        formatter.logger.info("  ⚙️  Executando backward pass...")
        loss = output.sum()
        loss.backward()

        formatter.print_tensor_info("Output (após backward)", output)
        formatter.print_tensor_info("Input (x) (após backward)", x)

        assert x.grad is not None
        assert linear_layer.weights.grad is not None
        assert linear_layer.bias.grad is not None

        formatter.logger.info(f"  🔹 Parâmetros da Camada Linear:")
        formatter.print_tensor_info("  Weights", linear_layer.weights)
        formatter.print_tensor_info("  Bias", linear_layer.bias)
        formatter.logger.info("  ✅ Teste da camada linear concluído.")

    def test_sigmoid_activation(self, formatter):
        formatter.print_section_header("Teste da Ativação Sigmoid")
        sigmoid = Sigmoid()
        x = Tensor([[-1.0, 0.0, 2.0]], requires_grad=True)
        formatter.print_tensor_info("Input (x)", x)

        output = sigmoid(x)

        flat_out = output._flatten(output.data)
        assert all(0.0 < val < 1.0 for val in flat_out)

        loss = output.sum()
        loss.backward()

        formatter.print_tensor_info("Output (após backward)", output)
        formatter.print_tensor_info("Input (x) (após backward)", x)

        assert x.grad is not None
        formatter.logger.info("  ✅ Teste da ativação Sigmoid concluído.")

    def test_relu_activation(self, formatter):
        formatter.print_section_header("Teste da Ativação ReLU")
        relu = ReLU()
        x = Tensor([[-1.0, 0.0, 2.0]], requires_grad=True)
        formatter.print_tensor_info("Input (x)", x)

        output = relu(x)

        flat_out = output._flatten(output.data)
        assert all(val >= 0 for val in flat_out)

        assert flat_out[0] == 0.0

        loss = output.sum()
        loss.backward()

        formatter.print_tensor_info("Output (após backward)", output)
        formatter.print_tensor_info("Input (x) (após backward)", x)

        flat_grad = x.grad._flatten(x.grad.data)
        assert flat_grad[0] == 0.0
        assert flat_grad[2] == 1.0

        formatter.logger.info("  ✅ Teste da ativação ReLU concluído.")

    def test_model_integration(self, formatter):
        formatter.print_section_header("Teste de Integração: Rede Neural Simples")

        class SimpleNet(Module):
            def __init__(self):
                super().__init__()
                self.add_module("fc1", Linear(in_features=3, out_features=4))
                self.add_module("activation", ReLU())
                self.add_module("fc2", Linear(in_features=4, out_features=1))

            def forward(self, x):
                x = self.fc1(x)
                x = self.activation(x)
                x = self.fc2(x)
                return x

        model = SimpleNet()
        formatter.logger.info("  🔹 Modelo 'SimpleNet' criado.")

        params = model.parameters()
        formatter.logger.info(
            f"  🔹 Número de tensores de parâmetros encontrados: {len(params)}"
        )

        assert len(params) == 4

        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        formatter.logger.info("  🔹 Gradientes calculados para todos os parâmetros.")

        for i, p in enumerate(params):
            formatter.print_tensor_info(f"  Param {i+1}", p)
            assert p.grad is not None

        model.zero_grad()
        formatter.logger.info(
            "  🔹 Gradientes zerados com sucesso usando `zero_grad()`."
        )

        all_grads_zero = all(
            p.grad is not None and all(g == 0.0 for g in p.grad._flatten(p.grad.data))
            for p in params
        )

        if all_grads_zero:
            formatter.logger.info("  ✅ Verificação `zero_grad` bem-sucedida.")
        else:
            formatter.logger.error("  ❌ Falha na verificação de `zero_grad`.")

        assert all_grads_zero, "Os gradientes não foram zerados corretamente."

        formatter.logger.info("  ✅ Teste de integração concluído.")
