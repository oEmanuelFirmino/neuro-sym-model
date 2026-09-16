"""Operadores fuzzy batched, espelhando exatamente as fórmulas de
`src/neurosym/interpreter/fuzzy_operators.py` (mesmo motor lógico, só trocando
`Tensor`/escalar por `torch.Tensor`/lote -- os operandos aqui têm shape `(B,)`)."""

from typing import Callable, Dict

import torch

FuzzyOperator = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def product_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b


def product_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b - (a * b)


def product_implies(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - a + (a * b)


def lukasiewicz_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.relu(a + b - 1.0)


def lukasiewicz_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.relu((1.0 - a) + (1.0 - b) - 1.0)


def lukasiewicz_implies(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.relu(a - b)


def logical_not(a: torch.Tensor) -> torch.Tensor:
    return 1.0 - a


OPERATOR_MAP: Dict[str, FuzzyOperator] = {
    "product_and": product_and,
    "product_or": product_or,
    "product_implies": product_implies,
    "lukasiewicz_and": lukasiewicz_and,
    "lukasiewicz_or": lukasiewicz_or,
    "lukasiewicz_implies": lukasiewicz_implies,
}

DEFAULT_OPERATORS = {
    "and": "product_and",
    "or": "product_or",
    "implies": "product_implies",
}


def get_operator(name: str) -> FuzzyOperator:
    if name not in OPERATOR_MAP:
        raise ValueError(
            f"Operador fuzzy desconhecido: '{name}'. Os operadores disponíveis são: {list(OPERATOR_MAP.keys())}"
        )
    return OPERATOR_MAP[name]
