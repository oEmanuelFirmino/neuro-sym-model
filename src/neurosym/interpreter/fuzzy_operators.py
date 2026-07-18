from typing import Dict, Callable
from src.neurosym.tensor import Tensor

FuzzyOperator = Callable[[Tensor, Tensor], Tensor]


def product_tnorm(a: Tensor, b: Tensor) -> Tensor:
    return a * b


def product_tconorm(a: Tensor, b: Tensor) -> Tensor:
    return a + b - (a * b)


def product_implication(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(1.0) - a + (a * b)


def _relu01(x: Tensor) -> Tensor:
    """max(0, x), implementado via relu() (Tensor não tem clamp superior;
    não é necessário aqui pois os operandos de Lukasiewicz já ficam em [0,1])."""
    return x.relu()


def lukasiewicz_tnorm(a: Tensor, b: Tensor) -> Tensor:
    # max(0, a + b - 1)
    return _relu01(a + b - Tensor(1.0))


def lukasiewicz_tconorm(a: Tensor, b: Tensor) -> Tensor:
    # min(1, a + b) == 1 - max(0, (1-a) + (1-b) - 1)
    return Tensor(1.0) - _relu01((Tensor(1.0) - a) + (Tensor(1.0) - b) - Tensor(1.0))


def lukasiewicz_implication(a: Tensor, b: Tensor) -> Tensor:
    # min(1, 1 - a + b) == 1 - max(0, a - b)
    return Tensor(1.0) - _relu01(a - b)


OPERATOR_MAP: Dict[str, FuzzyOperator] = {
    "product_and": product_tnorm,
    "product_or": product_tconorm,
    "product_implies": product_implication,
    "lukasiewicz_and": lukasiewicz_tnorm,
    "lukasiewicz_or": lukasiewicz_tconorm,
    "lukasiewicz_implies": lukasiewicz_implication,
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
