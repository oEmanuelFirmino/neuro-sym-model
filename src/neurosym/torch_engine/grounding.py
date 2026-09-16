from typing import Dict, Sequence

import torch
from torch import nn


class BatchedGroundingEnv(nn.Module):
    """Uma única matriz de embeddings compartilhada (num_constants, embedding_dim),
    substituindo o "um Tensor por constante" do motor original
    (`src/neurosym/interpreter`'s `grounding_env: Dict[str, Tensor]`). Índices de
    constantes são resolvidos uma vez, no momento da compilação das fórmulas
    (`torch_engine.compile`), não a cada avaliação -- é essa matriz compartilhada +
    resolução antecipada de índices que permite `table[idx]` virar um único gather
    vetorizado sobre um lote inteiro de fatos, em vez de um lookup de dicionário por
    fato."""

    def __init__(self, constant_names: Sequence[str], embedding_dim: int, seed: int = 0):
        super().__init__()
        self.vocab: Dict[str, int] = {name: i for i, name in enumerate(constant_names)}
        generator = torch.Generator().manual_seed(seed)
        table = torch.empty(len(constant_names), embedding_dim).uniform_(
            -1.0, 1.0, generator=generator
        )
        self.table = nn.Parameter(table)

    @property
    def embedding_dim(self) -> int:
        return self.table.shape[1]

    def indices_of(self, names: Sequence[str]) -> torch.Tensor:
        try:
            return torch.tensor([self.vocab[n] for n in names], dtype=torch.long)
        except KeyError as e:
            raise KeyError(f"Constante desconhecida no grounding: {e}") from e

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        return self.table[idx]
