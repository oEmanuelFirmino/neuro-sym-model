import sys
import logging
import pytest

try:
    from src.neurosym.tensor.tensor import Tensor
    from src.neurosym.module.module import Module, Linear, Sigmoid
    from src.neurosym.logic.logic import (
        Formula,
        Atom,
        Forall,
        Variable,
        Constant,
        Implies,
    )
    from src.neurosym.interpreter.interpreter import (
        Interpreter,
    )
except ImportError:
    pytest.fail(
        "❌ Erro ao importar módulos para o teste do interpretador.", pytrace=False
    )


class InterpreterTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger("InterpreterTest")
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
        self.logger.info(f"  🔗 {title.upper()} 🔗")
        self.logger.info("=" * 75)

    def print_section_header(self, section_name: str):
        self.logger.info("")
        self.logger.info(f"▶️  {section_name}")
        self.logger.info("<" + "-" * 60)

    def print_info(self, message: str):
        self.logger.info(f"  🔹 {message}")

    def print_tensor(self, name: str, tensor: Tensor):

        flat_data = tensor._flatten(tensor.data)
        data_str = f"[{', '.join([f'{x:.4f}' for x in flat_data])}]"
        self.logger.info(f"     Tensor '{name}': {data_str}")

    def print_result(self, formula: Formula, result: Tensor):
        self.logger.info(f"  🔹 Avaliando Fórmula: {formula}")
        scalar_value = result._flatten(result.data)[0]
        self.logger.info(f"     Resultado (Grau de Verdade): {scalar_value:.4f}")


class PredicateNet(Module):
    def __init__(self, in_features):
        super().__init__()
        self.add_module("layer", Linear(in_features, 1))
        self.add_module("activation", Sigmoid())

    def forward(self, x):
        return self.activation(self.layer(x))


@pytest.fixture
def formatter():
    return InterpreterTestFormatter()


@pytest.fixture
def env_setup(formatter):
    formatter.print_section_header("Configurando o Ambiente de Grounding (Fixture)")

    embedding_dim = 2

    grounding_env = {
        "socrates": Tensor([[1.0, 0.0]], requires_grad=True),
        "platao": Tensor([[0.9, 0.1]], requires_grad=True),
        "aristoteles": Tensor([[0.8, 0.2]], requires_grad=True),
    }

    for name, tensor in grounding_env.items():
        formatter.print_tensor(name, tensor)

    predicate_map = {
        "Mortal": PredicateNet(embedding_dim),
        "Homem": PredicateNet(embedding_dim),
        "Grego": PredicateNet(embedding_dim),
    }

    formatter.print_info("Mapa de Predicados criado com sucesso.")

    return grounding_env, predicate_map


class TestInterpreter:
    def test_atom_evaluation(self, env_setup, formatter):
        formatter.print_section_header("Testando a Avaliação de Átomos")

        grounding_env, predicate_map = env_setup
        interpreter = Interpreter(predicate_map, grounding_env)

        socrates = Constant("socrates")
        formula = Atom("Mortal", [socrates])

        result = interpreter.eval_formula(formula, {})

        formatter.print_result(formula, result)
        formatter.print_info("O resultado é um Tensor.")

        assert isinstance(result, Tensor), "O resultado da avaliação deve ser um Tensor"
        assert (
            result.requires_grad
        ), "O tensor resultante deve suportar gradiente (requires_grad=True)"
        assert result.shape == (1, 1) or result.shape == (
            1,
        ), f"Shape inesperado: {result.shape}"

    def test_complex_formula_evaluation(self, env_setup, formatter):
        formatter.print_section_header("Testando a Avaliação de Fórmulas Complexas")

        grounding_env, predicate_map = env_setup
        interpreter = Interpreter(predicate_map, grounding_env)

        x = Variable("x")

        axiom = Forall(x, Implies(Atom("Homem", [x]), Atom("Mortal", [x])))

        result = interpreter.eval_formula(axiom, {})
        formatter.print_result(axiom, result)

        assert isinstance(result, Tensor)

        formatter.print_info("Executando backward pass...")

        for t in grounding_env.values():
            t.zero_grad()

        result.backward()

        formatter.print_info("Backpropagation executado.")

        some_grad = False
        for name, const_tensor in grounding_env.items():
            if const_tensor.grad:

                flat_grad = const_tensor.grad._flatten(const_tensor.grad.data)
                if any(g != 0 for g in flat_grad):
                    some_grad = True
                    formatter.print_info(
                        f"Gradiente detectado em '{name}': {flat_grad}"
                    )

        if not some_grad:
            formatter.logger.warning(
                "Nenhum gradiente foi calculado para os embeddings."
            )

        assert (
            some_grad
        ), "Falha na retropropagação: Os gradientes dos embeddings permaneceram zerados ou None."
        formatter.print_info("✅ Verificação de gradientes concluída com sucesso.")
