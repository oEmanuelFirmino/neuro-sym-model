import math
from typing import Any, Tuple, Union


class Tensor:
    def __init__(self, data: Union[list, float, int]):
        if isinstance(data, (int, float)):
            self.data = data
            self.shape = ()
        elif isinstance(data, list):
            self._validate(data)
            self.data = data
            self.shape = self._get_shape(data)
        else:
            raise TypeError(
                "Tensor aceita apenas listas aninhadas ou numeros (int/float)."
            )

    def _validate(self, data: Any):
        if not isinstance(data, list):
            return
        first_shape = self._get_shape(data[0])
        for elem in data:
            if self._get_shape(elem) != first_shape:
                raise ValueError(
                    "Tensor mal formado: sublistas com dimensoes diferentes."
                )

    def _get_shape(self, data: Any) -> Tuple[int, ...]:
        if isinstance(data, list):
            if len(data) == 0:
                return (0,)
            return (len(data),) + self._get_shape(data[0])
        else:
            return ()

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"

    def _elementwise_op(self, other, op):
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError(
                    "Shapes incompativeis para operacao elemento a elemento."
                )
            return Tensor(self._apply_recursive(self.data, other.data, op))
        elif isinstance(other, (int, float)):
            return Tensor(self._apply_recursive(self.data, other, op))
        else:
            raise TypeError("Operacao nao suportada com tipo fornecido.")

    def _apply_recursive(self, a, b, op):
        if isinstance(a, list):
            return [
                self._apply_recursive(x, b if not isinstance(b, list) else y, op)
                for x, y in zip(a, b if isinstance(b, list) else [b] * len(a))
            ]
        else:
            return op(a, b)

    def __add__(self, other):
        return self._elementwise_op(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._elementwise_op(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._elementwise_op(other, lambda a, b: a * b)

    def dot(self, other: "Tensor") -> "Tensor":
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("Multiplicacao matricial exige tensores 2D.")
        if self.shape[1] != other.shape[0]:
            raise ValueError("Dimensoes incompativeis para multiplicacao matricial.")

        result = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                val = sum(
                    self.data[i][k] * other.data[k][j] for k in range(self.shape[1])
                )
                row.append(val)
            result.append(row)
        return Tensor(result)

    def exp(self):
        return Tensor(self._apply_recursive(self.data, None, lambda a, _: math.exp(a)))

    def log(self):
        return Tensor(self._apply_recursive(self.data, None, lambda a, _: math.log(a)))

    def pow(self, p: float):
        return Tensor(self._apply_recursive(self.data, None, lambda a, _: a**p))

    def _flatten(self, data):
        if isinstance(data, list):
            res = []
            for x in data:
                res.extend(self._flatten(x))
            return res
        else:
            return [data]

    def sum(self):
        return sum(self._flatten(self.data))

    def mean(self):
        flat = self._flatten(self.data)
        return sum(flat) / len(flat)


if __name__ == "__main__":
    t1 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    t2 = Tensor([[5.0, 6.0], [7.0, 8.0]])

    print(t1 + t2)
    print(t1 * 2)
    print(t1.dot(t2))

    print(t1.exp())
    print(t1.log())
    print(t1.pow(2))

    print(t1.sum())
    print(t1.mean())
