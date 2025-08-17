import math
from typing import Any, Tuple, Union, Iterable, Optional

Number = Union[int, float]


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float))


class Tensor:
    def __init__(
        self,
        data: Union[list, Number],
        requires_grad: bool = False,
        _op: Optional[str] = None,
        _parents: Optional[Iterable["Tensor"]] = None,
    ):
        if _is_number(data):
            self.data = float(data)
            self.shape = ()
        elif isinstance(data, list):
            self._validate(data)
            self.data = data
            self.shape = self._get_shape(data)
        else:
            raise TypeError(
                "Tensor aceita apenas listas aninhadas ou números (int/float)."
            )

        self.requires_grad = bool(requires_grad)
        self.grad: Optional[Union[list, float]] = None
        self._op: Optional[str] = _op
        self._parents: Tuple["Tensor", ...] = tuple(_parents) if _parents else tuple()
        self._backward = lambda: None

    def _validate(self, data: Any):
        if not isinstance(data, list):
            return
        if len(data) == 0:
            return
        first_shape = self._get_shape(data[0])
        for elem in data:
            if self._get_shape(elem) != first_shape:
                raise ValueError(
                    "Tensor mal formado: sublistas com dimensões diferentes."
                )

    def _get_shape(self, data: Any) -> Tuple[int, ...]:
        if isinstance(data, list):
            if len(data) == 0:
                return (0,)
            return (len(data),) + self._get_shape(data[0])
        else:
            return ()

    def __repr__(self):
        rg = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor(shape={self.shape}{rg}, data={self.data})"

    @staticmethod
    def _zeros_like_data(data: Union[list, float]) -> Union[list, float]:
        if isinstance(data, list):
            return [Tensor._zeros_like_data(x) for x in data]
        return 0.0

    def zero_grad(self):
        self.grad = self._zeros_like_data(self.data)

    def detach(self) -> "Tensor":
        return Tensor(self._clone_data(self.data), requires_grad=False)

    @staticmethod
    def _clone_data(data: Union[list, float]) -> Union[list, float]:
        if isinstance(data, list):
            return [Tensor._clone_data(x) for x in data]
        return float(data)

    @staticmethod
    def _flatten(data):
        if isinstance(data, list):
            out = []
            for x in data:
                out.extend(Tensor._flatten(x))
            return out
        return [data]

    def _apply_recursive(self, a, b, op):
        if isinstance(a, list):
            if isinstance(b, list):
                return [self._apply_recursive(x, y, op) for x, y in zip(a, b)]
            else:
                return [self._apply_recursive(x, b, op) for x in a]
        else:
            return op(a, b)

    def _ensure_same_shape(self, other: "Tensor"):
        if self.shape != other.shape:
            raise ValueError("Shapes incompatíveis para operação elemento a elemento.")

    def _wrap_result(
        self,
        data,
        op_name: str,
        parents: Iterable["Tensor"],
        requires_grad: Optional[bool] = None,
    ):
        req = (
            any(p.requires_grad for p in parents)
            if requires_grad is None
            else requires_grad
        )
        return Tensor(data, requires_grad=req, _op=op_name, _parents=parents)

    def _elementwise_op(self, other, op_name, op_fn):
        if isinstance(other, Tensor):
            self._ensure_same_shape(other)
            out_data = self._apply_recursive(self.data, other.data, op_fn)
            return self._wrap_result(out_data, op_name, (self, other))
        elif _is_number(other):
            out_data = self._apply_recursive(self.data, float(other), op_fn)
            return self._wrap_result(out_data, op_name, (self,))
        else:
            raise TypeError("Operação não suportada com tipo fornecido.")

    def __add__(self, other):
        return self._elementwise_op(other, "add", lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._elementwise_op(other, "sub", lambda a, b: a - b)

    def __rsub__(self, other):
        if _is_number(other):
            return self._elementwise_op(other, "rsub", lambda a, b: b - a)
        elif isinstance(other, Tensor):
            return other.__sub__(self)
        raise TypeError("Operação não suportada com tipo fornecido.")

    def __mul__(self, other):
        return self._elementwise_op(other, "mul", lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def dot(self, other: "Tensor") -> "Tensor":
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("Multiplicação matricial exige tensores 2D.")
        if self.shape[1] != other.shape[0]:
            raise ValueError("Dimensões incompatíveis para multiplicação matricial.")

        result = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                val = 0.0
                for k in range(self.shape[1]):
                    val += self.data[i][k] * other.data[k][j]
                row.append(val)
            result.append(row)
        return self._wrap_result(result, "dot", (self, other))

    def exp(self):
        out = self._apply_recursive(self.data, None, lambda a, _: math.exp(a))
        return self._wrap_result(out, "exp", (self,))

    def log(self):
        out = self._apply_recursive(self.data, None, lambda a, _: math.log(a))
        return self._wrap_result(out, "log", (self,))

    def pow(self, p: float):
        out = self._apply_recursive(self.data, None, lambda a, _: a**p)
        return self._wrap_result(out, f"pow({p})", (self,))

    def sum(self):
        return sum(self._flatten(self.data))

    def mean(self):
        flat = self._flatten(self.data)
        return sum(flat) / (len(flat) if flat else 1.0)

    def backward(self, grad: Optional[Union[list, float]] = None):
        raise NotImplementedError("Backward será implementado na Fase 2.2.")
