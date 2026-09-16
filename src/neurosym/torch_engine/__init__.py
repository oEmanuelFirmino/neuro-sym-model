"""Motor batched/GPU-capaz, ADITIVO ao motor original (`src/neurosym/tensor`,
`module`, `interpreter`, `training`) -- não o substitui nem o modifica. Existe
para validar a abordagem DLG em escala (hidden/embedding_dim grandes, milhares+
de fatos/entidades), onde o motor original (batch size 1, autograd sobre listas
Python) é dominado por overhead de laço Python, não por FLOPs.

Requer o extra opcional `torch` (`uv sync --extra torch`), por isso o import de
`torch` é adiado e verificado aqui com uma mensagem de erro clara, em vez de
quebrar o import de `src.neurosym` inteiro para quem não instalou a extra.
"""

try:
    import torch  # noqa: F401
except ImportError as e:
    raise ImportError(
        "src.neurosym.torch_engine requer PyTorch, que não é uma dependência "
        "obrigatória do projeto. Instale com `uv sync --extra torch` antes de "
        "importar este pacote."
    ) from e

from .compile import CompiledGroup, compile_formulas, leaves, signature
from .grounding import BatchedGroundingEnv
from .interpreter import BatchedInterpreter
from .model import DLGModel
from .modules import build_predicate_mlp, l1_penalty
from .saver import load_model, save_model
from .trainer import BatchedTrainer

__all__ = [
    "BatchedGroundingEnv",
    "BatchedInterpreter",
    "BatchedTrainer",
    "CompiledGroup",
    "DLGModel",
    "build_predicate_mlp",
    "compile_formulas",
    "l1_penalty",
    "leaves",
    "load_model",
    "save_model",
    "signature",
]
