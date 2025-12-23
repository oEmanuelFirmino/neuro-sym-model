import numpy as np
from typing import Any, Callable, Tuple
from .base import TensorBackend


class NumpyBackend(TensorBackend):
    def convert_to_internal(self, data: Any) -> np.ndarray:
        # Garante que seja array float64. Se já for, o overhead é mínimo.
        # copy=False evita duplicar memória se o array já for do tipo correto.
        if isinstance(data, np.ndarray):
            return data.astype(np.float64, copy=False)
        return np.array(data, dtype=np.float64)

    def get_shape(self, data: np.ndarray) -> Tuple[int, ...]:
        return data.shape

    def apply_elementwise(self, a: Any, b: Any, op: Callable) -> np.ndarray:
        # NumPy lida nativamente com 'op(array, array)' via operator overloading
        # e broadcasting automático.
        if b is None:
            return op(a)
        return op(a, b)

    def unbroadcast(
        self, data: np.ndarray, target_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Realiza a soma nas dimensões extras ou expandidas criadas pelo broadcasting.
        Isso substitui o loop recursivo lento da implementação anterior.
        """
        if data.shape == target_shape:
            return data

        ndims_added = len(data.shape) - len(target_shape)

        # 1. Soma nas dimensões extras à esquerda (ex: broadcasting de escalar para matriz)
        for _ in range(ndims_added):
            data = data.sum(axis=0)

        # 2. Se as dimensões existem mas têm tamanho 1 no target (broadcast), soma nelas.
        # Ex: target=(3, 1), current=(3, 5) -> soma no axis 1 para voltar a ter dimensão 1.
        for i, (dim_in, dim_out) in enumerate(zip(data.shape, target_shape)):
            if dim_in > dim_out:
                data = data.sum(axis=i, keepdims=True)

        return data

    def exp(self, data: Any) -> np.ndarray:
        return np.exp(data)

    def relu(self, data: Any) -> np.ndarray:
        return np.maximum(0, data)

    def relu_backward(self, grad_data: Any, input_data: Any) -> np.ndarray:
        # Operação vetorizada pura, sem loops.
        # Onde input > 0, passa o gradiente; caso contrário, zera.
        return np.where(input_data > 0, grad_data, 0.0)

    def dot(self, a: Any, b: Any) -> np.ndarray:
        return np.dot(a, b)

    def transpose(self, data: Any) -> np.ndarray:
        return data.T

    def sum(self, data: Any) -> float:
        return float(np.sum(data))

    def mean(self, data: Any) -> float:
        return float(np.mean(data))

    def p_mean(self, data: Any, p: float) -> float:
        # Implementação numericamente mais estável
        if data.size == 0:
            return 0.0
        # Adiciona 1e-9 para evitar log(0) ou divisões instáveis durante o backward
        val = np.mean((data + 1e-9) ** p) ** (1 / p)
        return float(val)

    def min(self, data: Any) -> float:
        if data.size == 0:
            return float("inf")
        return float(np.min(data))
