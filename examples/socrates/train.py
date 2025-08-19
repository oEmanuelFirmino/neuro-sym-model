import sys
import yaml
import logging
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.tensor import Tensor
    from src.module import Module, Linear, Sigmoid
    from src.interpreter import Interpreter
    from src.training.optimizer import SGD
    from src.data_manager.loader import KnowledgeBaseLoader
    from src.training.saver import save_model
    from src.training.trainer import Trainer
except ImportError as e:
    print(f"❌ Erro ao importar a biblioteca neuro-simbólica: {e}")
    sys.exit(1)

logger = logging.getLogger("SocratesExampleClient")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(
        description="Script cliente para treinar o modelo de Sócrates."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/socrates/config.yaml",
        help="Caminho para o arquivo de configuração YAML.",
    )
    args = parser.parse_args()

    logger.info("=" * 75)
    logger.info("  🚀 INICIANDO CLIENTE DE TREINAMENTO: O PROBLEMA DE SÓCRATES 🚀")
    logger.info("=" * 75)

    config_path = PROJECT_ROOT / args.config
    logger.info(f"Carregando configuração de: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_path = PROJECT_ROOT / config["data_path"]
    loader = KnowledgeBaseLoader(data_path)
    grounding_env = loader.load_domain(config["domain_file"], config["embedding_dim"])
    facts_with_truth_values = loader.load_facts(config["facts_file"])
    rules = loader.load_rules(config["rules_file"])
    test_facts = loader.load_facts(config["test_facts_file"])

    logger.info(
        f"Base de conhecimento carregada: {len(grounding_env)} constantes, {len(facts_with_truth_values)} fatos, {len(rules)} regras."
    )

    class PredicateNet(Module):
        def __init__(self, in_features):
            super().__init__()
            self.add_module("layer", Linear(in_features, 1))
            self.add_module("activation", Sigmoid())

        def forward(self, x):
            return self.activation(self.layer(x))

    predicate_map = {
        pred_config["name"]: PredicateNet(
            config["embedding_dim"] * pred_config["arity"]
        )
        for pred_config in config["predicates"]
    }
    logger.info(f"{len(predicate_map)} modelos de predicado foram instanciados.")

    all_parameters = list(grounding_env.values())
    for model in predicate_map.values():
        all_parameters.extend(model.parameters())
    optimizer = SGD(all_parameters, lr=config["learning_rate"])
    logger.info(
        f"Otimizador SGD configurado com learning rate: {config['learning_rate']}."
    )

    operator_config = config.get("fuzzy_logic")
    quantifier_config = config.get("quantifiers")

    interpreter = Interpreter(
        predicate_map=predicate_map,
        grounding_env=grounding_env,
        operator_config=operator_config,
        quantifier_config=quantifier_config,
    )
    logger.info(
        f"Interpretador instanciado com agregador Forall: {quantifier_config.get('forall') if quantifier_config else 'padrão'}."
    )

    trainer = Trainer(
        interpreter=interpreter, optimizer=optimizer, epochs=config["epochs"]
    )

    trainer.fit(rules=rules, facts=facts_with_truth_values)

    logger.info("\n" + "--- Avaliando o Modelo no Conjunto de Teste ---")
    for fact_formula, _ in test_facts:
        predicted_truth_tensor = interpreter.eval_formula(fact_formula, {})
        predicted_truth = predicted_truth_tensor._flatten(predicted_truth_tensor.data)[
            0
        ]
        logger.info(
            f"Consulta: {fact_formula}, Grau de Verdade Previsto: {predicted_truth:.4f}"
        )

    model_save_path = PROJECT_ROOT / config["model_save_path"]
    save_model(model_save_path, predicate_map, grounding_env)
    logger.info(f"Modelo salvo em: {model_save_path}")


if __name__ == "__main__":
    main()
