# examples/socrates/train.py

import sys
import yaml
import logging
import argparse
from pathlib import Path

from src.neurosym.tensor.backend import set_backend

set_backend("numpy")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.neurosym.interpreter import Interpreter
    from src.neurosym.training.optimizer import SGD
    from src.neurosym.data_manager.loader import KnowledgeBaseLoader
    from src.neurosym.training.trainer import Trainer
    from src.neurosym.module.factory import create_model_from_config
    from src.neurosym.training.callbacks import ModelCheckpoint
except ImportError as e:
    print(f"❌ Erro ao importar a biblioteca neuro-simbólica: {e}")
    sys.exit(1)

# Nome do logger alterado para ser mais genérico
logger = logging.getLogger("ExampleClient")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(
        description="Script cliente para treinar um modelo neuro-simbólico."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/socrates/config.yaml",
        help="Caminho para o arquivo YAML.",
    )
    args = parser.parse_args()

    logger.info("=" * 75)
    logger.info("  🚀 INICIANDO CLIENTE DE TREINAMENTO 🚀")
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

    logger.info(
        f"Base de conhecimento carregada: {len(grounding_env)} constantes, {len(facts_with_truth_values)} fatos, {len(rules)} regras."
    )

    predicate_map = {
        p["name"]: create_model_from_config(
            config["embedding_dim"] * p["arity"], p["architecture"]
        )
        for p in config["predicates"]
    }
    logger.info(
        f"{len(predicate_map)} modelos de predicado foram construídos dinamicamente."
    )

    all_parameters = list(grounding_env.values())
    for model in predicate_map.values():
        all_parameters.extend(model.parameters())
    optimizer = SGD(all_parameters, lr=config["learning_rate"])

    interpreter = Interpreter(
        predicate_map=predicate_map,
        grounding_env=grounding_env,
        operator_config=config.get("fuzzy_logic"),
        quantifier_config=config.get("quantifiers"),
    )

    model_save_path = PROJECT_ROOT / config["model_save_path"]
    checkpoint_callback = ModelCheckpoint(
        filepath=str(model_save_path), monitor="loss", mode="min"
    )

    trainer = Trainer(
        interpreter=interpreter,
        optimizer=optimizer,
        epochs=config["epochs"],
        callbacks=[checkpoint_callback],
    )

    trainer.fit(rules=rules, facts=facts_with_truth_values)

    # (CORREÇÃO) Torna a avaliação de teste opcional, verificando se a chave existe e não está vazia.
    test_facts_file = config.get("test_facts_file")
    if test_facts_file:
        logger.info("\n" + "--- Avaliando o Modelo no Conjunto de Teste ---")
        try:
            load_model(str(model_save_path), predicate_map, grounding_env)
            logger.info("Melhor modelo carregado para avaliação final.")

            test_facts = loader.load_facts(test_facts_file)
            for fact_formula, _ in test_facts:
                predicted_truth_tensor = interpreter.eval_formula(fact_formula, {})
                predicted_truth = predicted_truth_tensor._flatten(
                    predicted_truth_tensor.data
                )[0]
                logger.info(
                    f"Consulta: {fact_formula}, Grau de Verdade Previsto: {predicted_truth:.4f}"
                )
        except FileNotFoundError:
            logger.warning(
                "Nenhum ficheiro de teste encontrado ou modelo salvo pelo checkpoint. Avaliação ignorada."
            )
    else:
        logger.info(
            "\nNenhum 'test_facts_file' especificado na configuração. Avaliação final ignorada."
        )


if __name__ == "__main__":
    main()
