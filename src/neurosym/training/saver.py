import json
from pathlib import Path
from typing import Dict, Any, Union

from src.neurosym.module import Module
from src.neurosym.tensor import Tensor


def _make_serializable(obj: Any) -> Any:
    """
    Converte recursivamente objetos não serializáveis (como np.ndarray)
    em tipos nativos do Python (listas, floats, ints).
    """
    # Verifica se é um array numpy sem importar numpy forçadamente (evita erro se não instalado)
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return obj.tolist()
    
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(x) for x in obj]
    
    # Converte tipos numéricos numpy (np.float32, np.int64) para nativos
    if hasattr(obj, "item") and callable(obj.item) and hasattr(obj, "dtype"):
         return obj.item()

    return obj


def save_model(
    filepath: str, 
    predicate_map: Dict[str, Module], 
    grounding_env: Dict[str, Tensor]
):
    # Extrai o estado bruto (pode conter np.ndarrays)
    raw_state = {
        "predicates": {
            name: module.state_dict() for name, module in predicate_map.items()
        },
        "embeddings": {
            name: tensor.data for name, tensor in grounding_env.items()
        },
    }

    # Converte para formato serializável (listas)
    serializable_state = _make_serializable(raw_state)

    with open(filepath, "w") as f:
        json.dump(serializable_state, f, indent=2)


def load_model(
    filepath: str, 
    predicate_map: Dict[str, Module], 
    grounding_env: Dict[str, Tensor]
):
    with open(filepath, "r") as f:
        state = json.load(f)

    # Carrega pesos dos predicados
    if "predicates" in state:
        for name, module_state in state["predicates"].items():
            if name in predicate_map:
                predicate_map[name].load_state_dict(module_state)

    # Carrega embeddings do domínio
    if "embeddings" in state:
        for name, data in state["embeddings"].items():
            if name in grounding_env:
                # O Tensor se encarrega de converter a lista de volta para o backend atual
                grounding_env[name].data = Tensor(data).data