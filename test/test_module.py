import sys
import logging

try:
    from src.neurosym.tensor.tensor import Tensor
    from src.neurosym.module.module import Module, Linear, Sigmoid, ReLU
except ImportError:
    print("❌ Erro ao importar módulos.")
    print(
        "💡 Certifique-se de que os arquivos 'tensor.py' e 'module.py' estão nos diretórios corretos."
    )
    sys.exit(1)


class ModuleTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = self._setup_logger(log_level)

    def _setup_logger(self, log_level):
        logger = logging.getLogger("ModuleTest")
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


class ModuleTestSuite:
    def __init__(self):
        self.formatter = ModuleTestFormatter()

    def test_linear_layer(self):
        self.formatter.print_section_header("Teste da Camada Linear (Linear Layer)")

        in_features, out_features = 3, 2
        linear_layer = Linear(in_features, out_features)

        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        self.formatter.print_tensor_info("Input (x)", x)

        self.formatter.logger.info("  ⚙️  Executando forward pass...")
        output = linear_layer(x)
        self.formatter.print_tensor_info("Output", output)

        self.formatter.logger.info("  ⚙️  Executando backward pass...")
        loss = output.sum()
        loss.backward()

        self.formatter.print_tensor_info("Output (após backward)", output)
        self.formatter.print_tensor_info("Input (x) (após backward)", x)

        self.formatter.logger.info(f"  🔹 Parâmetros da Camada Linear:")
        self.formatter.print_tensor_info("  Weights", linear_layer.weights)
        self.formatter.print_tensor_info("  Bias", linear_layer.bias)
        self.formatter.logger.info("  ✅ Teste da camada linear concluído.")

    def test_sigmoid_activation(self):
        self.formatter.print_section_header("Teste da Ativação Sigmoid")
        sigmoid = Sigmoid()
        x = Tensor([[-1.0, 0.0, 2.0]], requires_grad=True)
        self.formatter.print_tensor_info("Input (x)", x)

        output = sigmoid(x)
        loss = output.sum()
        loss.backward()

        self.formatter.print_tensor_info("Output (após backward)", output)
        self.formatter.print_tensor_info("Input (x) (após backward)", x)
        self.formatter.logger.info("  ✅ Teste da ativação Sigmoid concluído.")

    def test_relu_activation(self):
        self.formatter.print_section_header("Teste da Ativação ReLU")
        relu = ReLU()
        x = Tensor([[-1.0, 0.0, 2.0]], requires_grad=True)
        self.formatter.print_tensor_info("Input (x)", x)

        output = relu(x)
        loss = output.sum()
        loss.backward()

        self.formatter.print_tensor_info("Output (após backward)", output)
        self.formatter.print_tensor_info("Input (x) (após backward)", x)
        self.formatter.logger.info("  ✅ Teste da ativação ReLU concluído.")

    def test_model_integration(self):
        self.formatter.print_section_header("Teste de Integração: Rede Neural Simples")

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
        self.formatter.logger.info("  🔹 Modelo 'SimpleNet' criado.")

        params = model.parameters()
        self.formatter.logger.info(
            f"  🔹 Número de tensores de parâmetros encontrados: {len(params)}"
        )

        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        self.formatter.logger.info(
            "  🔹 Gradientes calculados para todos os parâmetros."
        )
        for i, p in enumerate(params):
            self.formatter.print_tensor_info(f"  Param {i+1}", p)

        model.zero_grad()
        self.formatter.logger.info(
            "  🔹 Gradientes zerados com sucesso usando `zero_grad()`."
        )

        all_grads_zero = all(
            p.grad is not None and all(g == 0.0 for g in p.grad._flatten(p.grad.data))
            for p in params
        )
        if all_grads_zero:
            self.formatter.logger.info("  ✅ Verificação `zero_grad` bem-sucedida.")
        else:
            self.formatter.logger.error("  ❌ Falha na verificação de `zero_grad`.")

        self.formatter.logger.info("  ✅ Teste de integração concluído.")

    def run_all(self):
        self.formatter.print_banner("Teste de Módulos de Rede Neural")
        self.test_linear_layer()
        self.test_sigmoid_activation()
        self.test_relu_activation()
        self.test_model_integration()
        self.formatter.logger.info("\n🎉 Todos os testes foram concluídos com sucesso!")


def main():
    try:
        suite = ModuleTestSuite()
        suite.run_all()
    except Exception as e:
        print(f"❌ Erro fatal durante a execução dos testes: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
