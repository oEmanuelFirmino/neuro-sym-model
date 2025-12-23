import math
from typing import Any, Callable, Tuple, List
from .base import TensorBackend


def _flatten(data: Any) -> list:
    if isinstance(data, list):
        return [item for sublist in data for item in _flatten(sublist)]
    return [data]


class PythonBackend(TensorBackend):
    def convert_to_internal(self, data: Any) -> Any:
        """
        No backend Python, os dados internos são as próprias listas ou números.
        Não há conversão necessária.
        """
        return data

    def get_shape(self, data: Any) -> Tuple[int, ...]:
        """
        Calcula o shape recursivamente (lógica movida do antigo Tensor).
        """
        if isinstance(data, list) and data:
            return (len(data),) + self.get_shape(data[0])
        return ()

    def apply_elementwise(self, a: Any, b: Any, op: Callable) -> Any:
        """
        Aplica a operação recursivamente em listas aninhadas.
        (Antigo apply_recursive)
        """
        if isinstance(a, list):
            if b is not None and isinstance(b, list):
                # Caso onde ambos são listas
                if (
                    len(a) > 0
                    and isinstance(a[0], list)
                    and len(b) > 0
                    and not isinstance(b[0], list)
                ):
                    # Broadcasting simples: a é matriz, b é vetor/linha
                    return [self.apply_elementwise(row, b, op) for row in a]

                # Element-wise padrão
                return [self.apply_elementwise(x, y, op) for x, y in zip(a, b)]
            else:
                # b é escalar ou None
                return [self.apply_elementwise(x, b, op) for x in a]
        else:
            if b is not None and isinstance(b, list):
                # a é escalar, b é lista (broadcast reverso)
                return [self.apply_elementwise(a, y, op) for y in b]
            else:
                # Ambos escalares
                return op(a, b) if b is not None else op(a)

    def unbroadcast(self, data: Any, target_shape: Tuple[int, ...]) -> Any:
        """
        Reduz as dimensões extras para ajustar o gradiente ao shape original.
        Lógica portada da classe Tensor original.
        """
        current_shape = self.get_shape(data)

        if current_shape == target_shape:
            return data

        if not target_shape:
            return self.sum(data)

        # Se tiver mais dimensões que o alvo (ex: batching), soma as dimensões externas
        if len(current_shape) > len(target_shape):
            if isinstance(data, list) and len(data) > 0:
                accumulated = data[0]
                # Soma todos os elementos da dimensão extra
                for i in range(1, len(data)):
                    accumulated = self.apply_elementwise(
                        accumulated, data[i], lambda a, b: a + b
                    )
                # Continua recursivamente até atingir o número de dimensões correto
                return self.unbroadcast(accumulated, target_shape)

        return data

    def exp(self, data: Any) -> Any:
        return self.apply_elementwise(data, None, lambda a: math.exp(a))

    def relu(self, data: Any) -> Any:
        return self.apply_elementwise(data, None, lambda a: max(0, a))

    def relu_backward(self, grad_data: Any, input_data: Any) -> Any:
        return self.apply_elementwise(
            input_data, grad_data, lambda a, b: b if a > 0 else 0
        )

    def dot(self, a: Any, b: Any) -> Any:
        # Nota: dot product de listas requer validação de dimensões compatíveis,
        # assumimos listas bem formadas aqui.
        if not isinstance(a, list) or not isinstance(b, list):
            # Fallback para multiplicação escalar se não forem matrizes
            return a * b

        # Detectar se é vetor ou matriz para implementar corretamente
        # Implementação simplificada para matriz 2D x matriz 2D
        try:
            a_rows = len(a)
            a_cols = len(a[0]) if isinstance(a[0], list) else 0
            b_rows = len(b)
            b_cols = len(b[0]) if isinstance(b[0], list) else 0

            if a_cols == 0:  # a é vetor 1D
                # Produto escalar simples
                return sum(x * y for x, y in zip(a, b))

            return [
                [sum(a[i][k] * b[k][j] for k in range(a_cols)) for j in range(b_cols)]
                for i in range(a_rows)
            ]
        except Exception:
            # Em caso de erro de estrutura, retorna None ou levanta erro
            raise ValueError(
                "Falha no dot product do PythonBackend: formatos incompatíveis."
            )

    def transpose(self, data: Any) -> Any:
        if not isinstance(data, list) or not data or not isinstance(data[0], list):
            return data  # Escalar ou vetor 1D não transpõe da mesma forma
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
