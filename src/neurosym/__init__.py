# src/neurosym/__init__.py

# Expor as classes principais diretamente no topo
from .tensor.tensor import Tensor
from .logic.logic import Formula, Atom, Forall
from .module.module import Module, Linear, ReLU, Sigmoid
from .interpreter.interpreter import Interpreter
from .training.trainer import Trainer

__version__ = "0.1.0"

__all__ = [
    "Tensor",
    "Formula", "Atom", "Forall",
    "Module", "Linear", "ReLU", "Sigmoid",
    "Interpreter",
    "Trainer"
]