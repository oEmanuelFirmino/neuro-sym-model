from typing import List, Dict, Any
from src.neurosym.module import Module, Linear, Sigmoid, ReLU, Sequential


LAYER_MAP = {
    "Linear": Linear,
    "Sigmoid": Sigmoid,
    "ReLU": ReLU,
}


def create_model_from_config(
    in_features: int, architecture_config: List[Dict[str, Any]]
) -> Module:
    layers = []
    current_in_features = in_features

    for layer_conf in architecture_config:
        layer_type = layer_conf.get("type")
        if not layer_type or layer_type not in LAYER_MAP:
            raise ValueError(
                f"Tipo de camada inválido ou não especificado: {layer_type}"
            )

        LayerClass = LAYER_MAP[layer_type]

        if layer_type == "Linear":
            out_features = layer_conf.get("out_features")
            if not out_features:
                raise ValueError("A camada 'Linear' requer 'out_features'.")

            layers.append(
                LayerClass(in_features=current_in_features, out_features=out_features)
            )
            current_in_features = out_features
        else:

            layers.append(LayerClass())

    if not layers:
        raise ValueError("A configuração da arquitetura não pode estar vazia.")

    return Sequential(*layers)
