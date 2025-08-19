import math
from typing import Any, Callable
from .base import TensorBackend


def _flatten(data: Any) -> list:
    if isinstance(data, list):
        return [item for sublist in data for item in _flatten(sublist)]
    return [data]


class PythonBackend(TensorBackend):
    def apply_recursive(self, a: Any, b: Any, op: Callable) -> Any:
        if isinstance(a, list):
            if b is not None and isinstance(b, list):
                if (
                    len(a) > 0
                    and isinstance(a[0], list)
                    and len(b) > 0
                    and not isinstance(b[0], list)
                ):
                    return [self.apply_recursive(row, b, op) for row in a]
                return [self.apply_recursive(x, y, op) for x, y in zip(a, b)]
            else:
                return [self.apply_recursive(x, b, op) for x in a]
        else:  # a é escalar
            if b is not None and isinstance(b, list):
                return [self.apply_recursive(a, y, op) for y in b]
            else:
                return op(a, b) if b is not None else op(a)

    def exp(self, data: Any) -> Any:
        return self.apply_recursive(data, None, lambda a: math.exp(a))

    def relu(self, data: Any) -> Any:
        return self.apply_recursive(data, None, lambda a: max(0, a))

    def dot(self, a: Any, b: Any) -> Any:
        a_rows, a_cols = len(a), len(a[0])
        b_rows, b_cols = len(b), len(b[0])
        return [
            [sum(a[i][k] * b[k][j] for k in range(a_cols)) for j in range(b_cols)]
            for i in range(a_rows)
        ]

    def transpose(self, data: Any) -> Any:
        rows, cols = len(data), len(data[0])
        return [[data[j][i] for j in range(rows)] for i in range(cols)]

    def sum(self, data: Any) -> float:
        return sum(_flatten(data))

    def mean(self, data: Any) -> float:
        flat_data = _flatten(data)
        return sum(flat_data) / len(flat_data) if flat_data else 0.0

    def p_mean(self, data: Any, p: float) -> float:
        flat_data = _flatten(data)
        n = len(flat_data)
        if n == 0:
            return 0.0
        sum_of_powers = sum(x**p for x in flat_data)
        return (sum_of_powers / n) ** (1 / p)

    def min(self, data: Any) -> float:
        flat_data = _flatten(data)
        return min(flat_data) if flat_data else float("inf")
