import sys
import yaml
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

try:
    from src.neurosym.interpreter import Interpreter
    from src.neurosym.data_manager import KnowledgeBaseLoader
    from src.neurosym.training.saver import load_model
    from src.neurosym.module.factory import create_model_from_config

    from src.neurosym.explainability.explainer import explain_inference
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
        default="examples/socrates/config.yaml",
        help="Caminho para o arquivo de configuração.",
    )

    parser.add_argument(
        "--explain",
        action="store_true",
        help="Ativa o modo de explicabilidade para a consulta.",
    )
    args = parser.parse_args()

    # --- CORREÇÃO: Padronização para Lowercase ---
    # Garante que 'Socrates' vire 'socrates' para bater com o domínio.
    args.query = args.query.lower()
    # ---------------------------------------------

    config_path = PROJECT_ROOT / args.config
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo de configuração não encontrado em '{config_path}'.")
        sys.exit(1)

    print(f"--- Carregando Arquitetura do Modelo de '{config_path}' ---")

    data_path = PROJECT_ROOT / config["data_path"]
    loader = KnowledgeBaseLoader(data_path)
    grounding_env = loader.load_domain(config["domain_file"], config["embedding_dim"])

    predicate_map = {
        p["name"]: create_model_from_config(
            config["embedding_dim"] * p["arity"], p["architecture"]
        )
        for p in config["predicates"]
    }

    model_path = PROJECT_ROOT / config["model_save_path"]
    print(f"--- Carregando Pesos e Embeddings de '{model_path}' ---")
    try:
        load_model(str(model_path), predicate_map, grounding_env)
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo de modelo não encontrado em '{model_path}'.")
        print("💡 Certifique-se de que o modelo foi treinado primeiro.")
        sys.exit(1)

    interpreter = Interpreter(
        predicate_map=predicate_map,
        grounding_env=grounding_env,
        operator_config=config.get("fuzzy_logic"),
        quantifier_config=config.get("quantifiers"),
    )

    try:
        parsed_query = (
            loader._parse_rule(args.query)
            if "forall" in args.query
            else loader._parse_formula_str(args.query, {})
        )
    except Exception as e:
        print(f"❌ Erro ao interpretar a consulta '{args.query}': {e}")
        sys.exit(1)

    print(f"\n--- Avaliando a Consulta ---")
    result = interpreter.eval_formula(parsed_query, {})

    # Compatibilidade com o novo Tensor/NumpyBackend
    # Se result.data for escalar numpy, _flatten pode não ser necessário,
    # mas mantemos a chamada para garantir compatibilidade com a implementação atual do Tensor.
    truth_value = result._flatten(result.data)[0]

    print(f"Fórmula: {parsed_query}")
    print(f"Grau de Verdade: {truth_value:.4f}")

    if args.explain:
        print("\n--- Relatório de Explicabilidade (Influência de Constantes) ---")
        try:
            influences = explain_inference(parsed_query, interpreter)
            if not influences:
                print(
                    "Não foi possível determinar a influência (o gradiente pode ser zero)."
                )
            else:
                for i, (name, magnitude) in enumerate(influences):
                    print(
                        f"{i+1}. Constante: '{name}', Magnitude da Influência (Gradiente): {magnitude:.6f}"
                    )
        except Exception as e:
            print(f"❌ Falha ao gerar explicação: {e}")


if __name__ == "__main__":
    main()
