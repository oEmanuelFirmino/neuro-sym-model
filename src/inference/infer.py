import sys
import yaml
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.tensor import Tensor
    from src.module import Module, Linear, Sigmoid
    from src.logic import Formula
    from src.interpreter import Interpreter
    from src.data.loader import KnowledgeBaseLoader
    from src.training.saver import load_model
except ImportError:
    print("❌ Erro ao importar um ou mais módulos necessários para a inferência.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Realiza inferência com um modelo neuro-simbólico treinado."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="A fórmula lógica a ser avaliada. Ex: 'Mortal(socrates)'",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Caminho para o arquivo de configuração.",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"--- Carregando Arquitetura do Modelo de '{config_path}' ---")

    data_path = PROJECT_ROOT / config["data_path"]
    loader = KnowledgeBaseLoader(data_path)
    grounding_env = loader.load_domain(config["domain_file"], config["embedding_dim"])

    class PredicateNet(Module):
        def __init__(self, in_features):
            super().__init__()
            self.add_module("layer", Linear(in_features, 1))
            self.add_module("activation", Sigmoid())

        def forward(self, x):
            return self.activation(self.layer(x))

    predicate_map = {}
    for pred_config in config["predicates"]:
        arity = pred_config["arity"]
        predicate_map[pred_config["name"]] = PredicateNet(
            config["embedding_dim"] * arity
        )

    model_path = PROJECT_ROOT / config["model_save_path"]
    print(f"--- Carregando Pesos e Embeddings de '{model_path}' ---")
    try:
        load_model(model_path, predicate_map, grounding_env)
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo de modelo não encontrado em '{model_path}'.")
        print(
            "💡 Certifique-se de que o modelo foi treinado primeiro executando 'train.py'."
        )
        sys.exit(1)

    interpreter = Interpreter(predicate_map, grounding_env)

    try:

        parsed_query = loader._parse_formula_str(args.query, {})
    except Exception as e:
        print(f"❌ Erro ao interpretar a consulta '{args.query}': {e}")
        sys.exit(1)

    print(f"\n--- Avaliando a Consulta ---")
    result = interpreter.eval_formula(parsed_query, {})
    truth_value = result._flatten(result.data)[0]

    print(f"Fórmula: {args.query}")
    print(f"Grau de Verdade: {truth_value:.4f}")


if __name__ == "__main__":
    main()
