from typing import Dict

from torch import nn

from .grounding import BatchedGroundingEnv


class DLGModel(nn.Module):
    """Contêiner único (grounding_env + predicados) como um só `nn.Module`, para
    ganhar de graça `.parameters()`, `.to(device)` e `.state_dict()`/
    `.load_state_dict()` recursivos sobre tudo -- ver `torch_engine/saver.py`."""

    def __init__(self, grounding_env: BatchedGroundingEnv, predicate_map: Dict[str, nn.Module]):
        super().__init__()
        self.grounding_env = grounding_env
        self.predicate_map = nn.ModuleDict(predicate_map)
