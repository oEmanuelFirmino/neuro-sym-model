"""Equivalente batched de `src/neurosym/training/saver.py`. Como `DLGModel` é um
único `nn.Module` (grounding_env + predicate_map como submódulos), seu
`state_dict()` já recursa por tudo sozinho -- não precisa do passeio manual por
JSON que o motor original faz."""

from pathlib import Path

import torch

from .model import DLGModel


def save_model(path: str, model: DLGModel) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: str, model: DLGModel) -> DLGModel:
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model
