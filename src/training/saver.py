import json
from pathlib import Path
from typing import Dict

from src.tensor import Tensor
from src.module import Module, Linear, Sigmoid
from src.interpreter import PredicateMap, GroundingEnv


def save_model(path: str, predicate_map: PredicateMap, grounding_env: GroundingEnv):
    model_state = {
        "predicate_map": {
            name: module.state_dict() for name, module in predicate_map.items()
        },
        "grounding_env": {name: tensor.data for name, tensor in grounding_env.items()},
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(model_state, f, indent=2)


def load_model(path: str, predicate_map: PredicateMap, grounding_env: GroundingEnv):
    with open(path, "r") as f:
        model_state = json.load(f)

    for name, state_dict in model_state["predicate_map"].items():
        if name in predicate_map:
            predicate_map[name].load_state_dict(state_dict)

    for name, data in model_state["grounding_env"].items():
        if name in grounding_env:
            grounding_env[name].data = data

    return predicate_map, grounding_env
