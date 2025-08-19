from typing import Dict, Callable
from src.tensor import Tensor

FuzzyOperator = Callable[[Tensor, Tensor], Tensor]


def product_tnorm(a: Tensor, b: Tensor) -> Tensor:
    return a * b


def product_tconorm(a: Tensor, b: Tensor) -> Tensor:
    return a + b - (a * b)


def product_implication(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(1.0) - a + (a * b)


OPERATOR_MAP: Dict[str, FuzzyOperator] = {
    "product_and": product_tnorm,
    "product_or": product_tconorm,
    "product_implies": product_implication,
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
