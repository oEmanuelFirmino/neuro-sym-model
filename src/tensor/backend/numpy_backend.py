import numpy as np
from typing import Any, Callable
from .base import TensorBackend


class NumpyBackend(TensorBackend):
    def _to_numpy(self, data: Any) -> np.ndarray:
        return np.array(data, dtype=np.float64)

    def _to_list(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            return data.tolist()
        return data.item() if hasattr(data, "item") else data

    def apply_recursive(self, a: Any, b: Any, op: Callable) -> Any:
        a_np = self._to_numpy(a)
        if b is not None:
            result = op(a_np, self._to_numpy(b))
        else:
            if op(None) == 0.0:
                result = np.zeros_like(a_np)
            else:
                result = np.vectorize(op)(a_np)
        return self._to_list(result)

    def exp(self, data: Any) -> Any:
        return self._to_list(np.exp(self._to_numpy(data)))

    def relu(self, data: Any) -> Any:
        return self._to_list(np.maximum(0, self._to_numpy(data)))

    def relu_backward(self, grad_data: Any, input_data: Any) -> Any:
        """(CORREÇÃO) Usa np.where para a derivada vetorial da ReLU."""
        grad_np = self._to_numpy(grad_data)
        input_np = self._to_numpy(input_data)
        return self._to_list(np.where(input_np > 0, grad_np, 0))

    def dot(self, a: Any, b: Any) -> Any:
        return self._to_list(np.dot(self._to_numpy(a), self._to_numpy(b)))

    def transpose(self, data: Any) -> Any:
        return self._to_list(self._to_numpy(data).T)

    def sum(self, data: Any) -> float:
        return float(np.sum(self._to_numpy(data)))

    def mean(self, data: Any) -> float:
        return float(np.mean(self._to_numpy(data)))

    def p_mean(self, data: Any, p: float) -> float:
        array = self._to_numpy(data)
        if array.size == 0:
            return 0.0
        return float(np.mean((array + 1e-9) ** p) ** (1 / p))

    def min(self, data: Any) -> float:
        array = self._to_numpy(data)
        return float(np.min(array)) if array.size > 0 else float("inf")
