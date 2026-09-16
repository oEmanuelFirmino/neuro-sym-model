from typing import List

import torch
from torch import nn


def build_predicate_mlp(
    in_features: int,
    hidden: int,
    num_hidden_layers: int = 2,
    out_features: int = 1,
) -> nn.Sequential:
    """Mesmo formato usado pelos predicados do motor original (Linear->ReLU
    repetido, Linear->Sigmoid final -- ver `experiments/modular_addition/run.py`'s
    `_build_predicate`), agora como `nn.Sequential` batched."""
    layers: List[nn.Module] = []
    current = in_features
    for _ in range(num_hidden_layers):
        layers += [nn.Linear(current, hidden), nn.ReLU()]
        current = hidden
    layers += [nn.Linear(current, out_features), nn.Sigmoid()]
    return nn.Sequential(*layers)


def l1_penalty(predicate_map: nn.ModuleDict) -> torch.Tensor:
    """Equivalente batched de `Module.l1_weight_parameters()`: soma ||W||_1 sobre
    as matrizes de peso de todo `nn.Linear` nos predicados, excluindo bias (mesmo
    critério do motor original)."""
    terms = [
        layer.weight.abs().sum()
        for model in predicate_map.values()
        for layer in model.modules()
        if isinstance(layer, nn.Linear)
    ]
    if not terms:
        return torch.tensor(0.0)
    return torch.stack(terms).sum()
