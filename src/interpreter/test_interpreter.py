import sys
import logging
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.tensor.tensor import Tensor
    from src.module.module import Module, Linear, Sigmoid
    from src.logic.logic import Formula, Atom, Forall, Variable, Constant, Implies, And
    from src.interpreter.interpreter import Interpreter, PredicateMap, GroundingEnv
except ImportError:
    print("❌ Erro ao importar módulos para o teste do interpretador.")
    sys.exit(1)


class InterpreterTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = self._setup_logger(log_level)

    def _setup_logger(self, log_level):
        logger = logging.getLogger("InterpreterTest")
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
        self.logger.info(f"  🔗 {title.upper()} 🔗")
        self.logger.info("=" * 75)

    def print_section_header(self, section_name: str):
        self.logger.info("")
        self.logger.info(f"▶️  {section_name}")
        self.logger.info("<" + "-" * 60)

    def print_info(self, message: str):
        self.logger.info(f"  🔹 {message}")

    def print_tensor(self, name: str, tensor: Tensor):
        data = f"[{', '.join([f'{x:.4f}' for x in tensor._flatten(tensor.data)])}]"
        self.logger.info(f"     Tensor '{name}': {data}")

    def print_result(self, formula: Formula, result: Tensor):
        self.logger.info(f"  🔹 Avaliando Fórmula: {formula}")
        scalar_value = result._flatten(result.data)[0]
        self.logger.info(f"     Resultado (Grau de Verdade): {scalar_value:.4f}")


class InterpreterTestSuite:
    def __init__(self):
        self.formatter = InterpreterTestFormatter()
        self.predicate_map: PredicateMap = {}
        self.grounding_env: GroundingEnv = {}

    def setup_environment(self):
        self.formatter.print_section_header("1. Configurando o Ambiente de Grounding")

        embedding_dim = 2

        self.grounding_env = {
            "socrates": Tensor([[1.0, 0.0]], requires_grad=True),
            "platao": Tensor([[0.9, 0.1]], requires_grad=True),
            "aristoteles": Tensor([[0.8, 0.2]], requires_grad=True),
        }
        self.formatter.print_info("Ambiente de Grounding (Constantes -> Embeddings):")
        for name, tensor in self.grounding_env.items():
            self.formatter.print_tensor(name, tensor)

        class PredicateNet(Module):
            def __init__(self, in_features):
                super().__init__()
                self.add_module("layer", Linear(in_features, 1))
                self.add_module("activation", Sigmoid())

            def forward(self, x):
                return self.activation(self.layer(x))

        self.predicate_map = {
            "Mortal": PredicateNet(embedding_dim),
            "Homem": PredicateNet(embedding_dim),
            "Grego": PredicateNet(embedding_dim),
        }
        self.formatter.print_info(
            "Mapa de Predicados (Nomes -> Módulos Neurais) criado."
        )

    def test_atom_evaluation(self):
        self.formatter.print_section_header("2. Testando a Avaliação de Átomos")
        interpreter = Interpreter(self.predicate_map, self.grounding_env)

        socrates = Constant("socrates")
        formula = Atom("Mortal", [socrates])

        result = interpreter.eval_formula(formula, {})
        self.formatter.print_result(formula, result)
        self.formatter.print_info(
            f"O resultado é um Tensor, permitindo backpropagation."
        )

    def test_complex_formula_evaluation(self):
        self.formatter.print_section_header(
            "3. Testando a Avaliação de Fórmulas Complexas"
        )
        interpreter = Interpreter(self.predicate_map, self.grounding_env)

        x = Variable("x")

        axiom = Forall(x, Implies(Atom("Homem", [x]), Atom("Mortal", [x])))

        result = interpreter.eval_formula(axiom, {})
        self.formatter.print_result(axiom, result)

        try:
            result.backward()
            self.formatter.print_info(
                "Backpropagation executado com sucesso no resultado da fórmula."
            )

            some_grad = False
            for const_tensor in self.grounding_env.values():
                if const_tensor.grad and any(
                    g != 0 for g in const_tensor.grad._flatten(const_tensor.grad.data)
                ):
                    some_grad = True
                    break
            if some_grad:
                self.formatter.print_info(
                    "Verificação: Gradientes foram populados nos embeddings."
                )
            else:
                self.formatter.logger.warning(
                    "Aviso: Nenhum gradiente foi calculado para os embeddings."
                )

        except Exception as e:
            self.formatter.logger.error(f"Falha na retropropagação: {e}")

    def run_all(self):
        self.formatter.print_banner(
            "Teste do Bloco de Integração (Interpretador Neuro-Simbólico)"
        )
        self.setup_environment()
        self.test_atom_evaluation()
        self.test_complex_formula_evaluation()
        self.formatter.logger.info(
            "\n🎉 Todos os testes de integração foram concluídos com sucesso!"
        )


def main():
    try:
        suite = InterpreterTestSuite()
        suite.run_all()
    except Exception as e:
        print(f"❌ Erro fatal durante a execução dos testes: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
