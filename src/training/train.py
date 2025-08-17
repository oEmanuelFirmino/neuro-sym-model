import sys
import logging
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.tensor import Tensor
    from src.module import Module, Linear, Sigmoid
    from src.logic import Formula, Atom, Forall, Variable, Constant, Implies
    from src.interpreter import Interpreter, PredicateMap, GroundingEnv
    from src.training.optimizer import SGD
except ImportError:
    print("❌ Erro ao importar um ou mais módulos necessários para o treinamento.")
    sys.exit(1)


class TrainingLogger:
    def __init__(self, log_level=logging.INFO):
        self.logger = self._setup_logger(log_level)

    def _setup_logger(self, log_level):
        logger = logging.getLogger("NeuroSymbolicTraining")
        logger.setLevel(log_level)
        if logger.hasHandlers():
            logger.handlers.clear()
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def print_banner(self, title: str):
        self.logger.info("")
        self.logger.info("=" * 75)
        self.logger.info(f"  🚀 {title.upper()} 🚀")
        self.logger.info("=" * 75)

    def print_section(self, title: str):
        self.logger.info("\n" + f"--- {title} " + "-" * (65 - len(title)))

    def log_epoch(self, epoch, total_epochs, loss, satisfaction):
        self.logger.info(
            f"Época [{epoch+1}/{total_epochs}], Perda: {loss:.4f}, Satisfação Global: {satisfaction:.4f}"
        )

    def log_final_results(self, grounding_env: GroundingEnv):
        self.logger.info("\n--- Resultados Finais ---")
        self.logger.info("Embeddings das constantes após o treinamento:")
        for name, tensor in grounding_env.items():
            flat_data = tensor._flatten(tensor.data)
            data_str = ", ".join([f"{x:.4f}" for x in flat_data])
            self.logger.info(f"  🔹 {name}: [{data_str}]")


def setup_knowledge_base() -> List[Formula]:
    x = Variable("x")

    axiom1 = Forall(x, Implies(Atom("Grego", [x]), Atom("Homem", [x])))
    axiom2 = Forall(x, Implies(Atom("Homem", [x]), Atom("Mortal", [x])))

    socrates = Constant("socrates")
    fact1 = Atom("Grego", [socrates])

    return [axiom1, axiom2, fact1]


def main():
    logger = TrainingLogger()
    logger.print_banner("Loop de Treinamento Neuro-Simbólico")

    logger.print_section("1. Inicializando Ambiente e Modelos")

    embedding_dim = 2
    grounding_env: GroundingEnv = {
        "socrates": Tensor([[0.1, 0.9]], requires_grad=True),
        "platao": Tensor([[0.2, 0.8]], requires_grad=True),
        "aristoteles": Tensor([[0.3, 0.7]], requires_grad=True),
    }

    class PredicateNet(Module):
        def __init__(self, in_features):
            super().__init__()
            self.add_module("layer", Linear(in_features, 1))
            self.add_module("activation", Sigmoid())

        def forward(self, x):
            return self.activation(self.layer(x))

    predicate_map: PredicateMap = {
        "Mortal": PredicateNet(embedding_dim),
        "Homem": PredicateNet(embedding_dim),
        "Grego": PredicateNet(embedding_dim),
    }

    all_parameters = []
    for tensor in grounding_env.values():
        all_parameters.append(tensor)
    for model in predicate_map.values():
        all_parameters.extend(model.parameters())

    logger.logger.info(f"Total de tensores para otimizar: {len(all_parameters)}")

    knowledge_base = setup_knowledge_base()
    interpreter = Interpreter(predicate_map, grounding_env)
    optimizer = SGD(all_parameters, lr=0.1)

    epochs = 100
    logger.print_section("2. Iniciando Treinamento")
    for epoch in range(epochs):
        optimizer.zero_grad()

        truth_values = []
        for formula in knowledge_base:
            truth_values.append(interpreter.eval_formula(formula, {}))

        total_satisfaction = truth_values[0]
        for i in range(1, len(truth_values)):
            total_satisfaction += truth_values[i]
        total_satisfaction /= Tensor(len(truth_values))

        loss = (Tensor(1.0) - total_satisfaction).sum()

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            satisfaction_value = total_satisfaction._flatten(total_satisfaction.data)[0]
            logger.log_epoch(epoch, epochs, loss.data, satisfaction_value)

    logger.log_final_results(grounding_env)
    logger.logger.info("\n🎉 Treinamento concluído com sucesso!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erro fatal durante o treinamento: {e}")
        import traceback

        traceback.print_exc()
