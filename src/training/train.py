import sys
import yaml
import logging
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.tensor import Tensor
    from src.module import Module, Linear, Sigmoid
    from src.logic import Formula
    from src.interpreter import Interpreter, PredicateMap, GroundingEnv
    from src.training.optimizer import SGD
    from src.data.loader import KnowledgeBaseLoader
    from src.training.saver import save_model
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


def main():
    logger = TrainingLogger()
    logger.print_banner("Loop de Treinamento Neuro-Simbólico a partir de Arquivos")

    config_path = PROJECT_ROOT / "config.yaml"
    logger.logger.info(f"Carregando configuração de: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.print_section("1. Carregando Base de Conhecimento")
    data_path = PROJECT_ROOT / config["data_path"]
    loader = KnowledgeBaseLoader(data_path)

    grounding_env = loader.load_domain(config["domain_file"], config["embedding_dim"])
    logger.logger.info(f"Domínio com {len(grounding_env)} constantes carregado.")

    class PredicateNet(Module):
        def __init__(self, in_features):
            super().__init__()
            self.add_module("layer", Linear(in_features, 1))
            self.add_module("activation", Sigmoid())

        def forward(self, x):
            return self.activation(self.layer(x))

    predicate_map: PredicateMap = {}
    for pred_config in config["predicates"]:
        arity = pred_config["arity"]
        predicate_map[pred_config["name"]] = PredicateNet(
            config["embedding_dim"] * arity
        )
    logger.logger.info(f"{len(predicate_map)} predicados neurais criados.")

    facts_with_truth_values = loader.load_facts(config["facts_file"])
    rules = loader.load_rules(config["rules_file"])
    logger.logger.info(
        f"{len(facts_with_truth_values)} fatos e {len(rules)} regras carregados."
    )

    all_parameters = list(grounding_env.values())
    for model in predicate_map.values():
        all_parameters.extend(model.parameters())

    interpreter = Interpreter(predicate_map, grounding_env)
    optimizer = SGD(all_parameters, lr=config["learning_rate"])
    logger.logger.info(f"Otimizador SGD configurado com lr={config['learning_rate']}.")

    epochs = config["epochs"]
    logger.print_section("2. Iniciando Treinamento")
    for epoch in range(epochs):
        optimizer.zero_grad()

        rule_truth_values = [interpreter.eval_formula(formula, {}) for formula in rules]

        fact_losses = []
        for fact_formula, truth_value in facts_with_truth_values:
            predicted_truth = interpreter.eval_formula(fact_formula, {}).sum()
            fact_losses.append((predicted_truth - Tensor(truth_value)) ** 2)

        total_satisfaction = sum(rule_truth_values, Tensor(0.0))
        total_fact_loss = sum(fact_losses, Tensor(0.0))

        loss = (Tensor(len(rules)) - total_satisfaction) + total_fact_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            avg_satisfaction = (total_satisfaction.data / len(rules)) if rules else 0.0
            logger.log_epoch(epoch, epochs, loss.data, avg_satisfaction)

    logger.log_final_results(grounding_env)

    logger.print_section("3. Salvando Modelo")
    model_save_path = PROJECT_ROOT / config["model_save_path"]
    save_model(model_save_path, predicate_map, grounding_env)
    logger.logger.info(f"Modelo salvo em: {model_save_path}")

    logger.logger.info("\n🎉 Treinamento concluído com sucesso!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erro fatal durante o treinamento: {e}")
        import traceback

        traceback.print_exc()
