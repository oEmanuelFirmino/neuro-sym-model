import math
from typing import Any, Tuple, Union, Iterable, Optional, Callable, List

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
            self.shape: Tuple[int, ...] = ()
        elif isinstance(data, list):
            self._validate(data)
            self.data = data
            self.shape = self._get_shape(data)
        else:
            raise TypeError(
                "Tensor aceita apenas listas aninhadas ou números (int/float)."
            )

        self.requires_grad = bool(requires_grad)
        self.grad: Optional[Tensor] = None
        if self.requires_grad:
            self.zero_grad()

        self._op: Optional[str] = _op
        self._parents: Tuple["Tensor", ...] = tuple(_parents) if _parents else tuple()
        self._backward: Callable[[], None] = lambda: None

    def _validate(self, data: Any):
        if not isinstance(data, list) or not data:
            return
        first_shape = self._get_shape(data[0])
        for elem in data[1:]:
            if self._get_shape(elem) != first_shape:
                raise ValueError(
                    "Tensor mal formado: sublistas com dimensões diferentes."
                )

    def _get_shape(self, data: Any) -> Tuple[int, ...]:
        if isinstance(data, list):
            if not data:
                return (0,)
            return (len(data),) + self._get_shape(data[0])
        return ()

    def __repr__(self):
        rg = ", requires_grad=True" if self.requires_grad else ""
        op = f", op={self._op}" if self._op else ""
        return f"Tensor(shape={self.shape}{rg}{op}, data={self.data})"

    @staticmethod
    def _apply_recursive(a, b, op):
        if isinstance(a, list):
            if isinstance(b, list):
                if (
                    len(a) > 0
                    and isinstance(a[0], list)
                    and len(b) > 0
                    and not isinstance(b[0], list)
                ):
                    return [Tensor._apply_recursive(row, b, op) for row in a]
                return [Tensor._apply_recursive(x, y, op) for x, y in zip(a, b)]
            else:
                return [Tensor._apply_recursive(x, b, op) for x in a]
        else:
            if isinstance(b, list):
                return [Tensor._apply_recursive(a, y, op) for y in b]
            else:
                return op(a, b) if b is not None else op(a)

    def _wrap_result(
        self,
        data,
        op_name: str,
        parents: Iterable["Tensor"],
    ) -> "Tensor":
        requires_grad = any(p.requires_grad for p in parents)
        return Tensor(data, requires_grad=requires_grad, _op=op_name, _parents=parents)

    def __add__(self, other):
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self._apply_recursive(
            self.data, other_tensor.data, lambda a, b: a + b
        )
        out = self._wrap_result(out_data, "add", (self, other_tensor))

        def _backward():
            if self.requires_grad:
                grad_to_add = out.grad.data
                if self.shape != out.shape:
                    if self.shape == ():
                        grad_to_add = sum(Tensor._flatten(grad_to_add))
                    else:
                        for _ in range(len(out.shape) - len(self.shape)):
                            grad_to_add = [sum(row) for row in grad_to_add]
                self.grad.data = self._apply_recursive(
                    self.grad.data, grad_to_add, lambda a, b: a + b
                )
            if other_tensor.requires_grad:
                grad_to_add = out.grad.data
                if other_tensor.shape != out.shape:
                    if other_tensor.shape == ():
                        grad_to_add = sum(Tensor._flatten(grad_to_add))
                    else:
                        for _ in range(len(out.shape) - len(other_tensor.shape)):
                            grad_to_add = [sum(row) for row in grad_to_add]
                other_tensor.grad.data = self._apply_recursive(
                    other_tensor.grad.data, grad_to_add, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def __mul__(self, other):
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self._apply_recursive(
            self.data, other_tensor.data, lambda a, b: a * b
        )
        out = self._wrap_result(out_data, "mul", (self, other_tensor))

        def _backward():
            if self.requires_grad:
                grad_data = self._apply_recursive(
                    other_tensor.data, out.grad.data, lambda a, b: a * b
                )
                if self.shape != out.shape:
                    if self.shape == ():
                        grad_data = sum(Tensor._flatten(grad_data))
                    else:
                        for _ in range(len(out.shape) - len(self.shape)):
                            grad_data = [sum(row) for row in grad_data]
                self.grad.data = self._apply_recursive(
                    self.grad.data, grad_data, lambda a, b: a + b
                )
            if other_tensor.requires_grad:
                grad_data = self._apply_recursive(
                    self.data, out.grad.data, lambda a, b: a * b
                )
                if other_tensor.shape != out.shape:
                    if other_tensor.shape == ():
                        grad_data = sum(Tensor._flatten(grad_data))
                    else:
                        for _ in range(len(out.shape) - len(other_tensor.shape)):
                            grad_data = [sum(row) for row in grad_data]
                other_tensor.grad.data = self._apply_recursive(
                    other_tensor.grad.data, grad_data, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def __pow__(self, p: float):
        if not _is_number(p):
            raise TypeError(
                "A potência só é suportada para expoentes numéricos (int/float)."
            )
        out_data = self._apply_recursive(self.data, None, lambda a: a**p)
        out = self._wrap_result(out_data, f"pow({p})", (self,))

        def _backward():
            if self.requires_grad:

                grad_data = self._apply_recursive(
                    self.data, out.grad.data, lambda a, b: (p * (a ** (p - 1))) * b
                )
                self.grad.data = self._apply_recursive(
                    self.grad.data, grad_data, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def sum(self) -> "Tensor":
        total = sum(self._flatten(self.data))
        out = self._wrap_result(total, "sum", (self,))

        def _backward():
            if self.requires_grad:
                grad_dist = out.grad.data
                self.grad.data = self._apply_recursive(
                    self.grad.data, None, lambda a: a + grad_dist
                )

        out._backward = _backward
        return out

    def mean(self) -> "Tensor":
        flat_data = self._flatten(self.data)
        n = len(flat_data)
        mean_val = sum(flat_data) / n if n > 0 else 0.0
        out = self._wrap_result(mean_val, "mean", (self,))

        def _backward():
            if self.requires_grad:
                grad_dist = out.grad.data / n
                self.grad.data = self._apply_recursive(
                    self.grad.data, None, lambda a: a + grad_dist
                )

        out._backward = _backward
        return out

    def p_mean(self, p: float) -> "Tensor":
        flat_data = self._flatten(self.data)
        n = len(flat_data)
        if n == 0:
            return self._wrap_result(0.0, f"p_mean({p})", (self,))

        sum_of_powers = sum(x**p for x in flat_data)
        p_mean_val = (sum_of_powers / n) ** (1 / p)
        out = self._wrap_result(p_mean_val, f"p_mean({p})", (self,))

        def _backward():
            if self.requires_grad:

                pre_factor = (out.data ** (1 - p)) / n

                def grad_fn(x, grad_out):
                    return (x ** (p - 1)) * pre_factor * grad_out

                grad_data = self._apply_recursive(self.data, out.grad.data, grad_fn)
                self.grad.data = self._apply_recursive(
                    self.grad.data, grad_data, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def min(self) -> "Tensor":
        flat_data = self._flatten(self.data)
        if not flat_data:
            return self._wrap_result(float("inf"), "min", (self,))
        min_val = min(flat_data)
        out = self._wrap_result(min_val, "min", (self,))

        def _backward():
            if self.requires_grad:
                min_indices = [i for i, x in enumerate(flat_data) if x == min_val]
                grad_data = self._apply_recursive(self.grad.data, None, lambda _: 0.0)
                flat_grad = self._flatten(grad_data)

                grad_per_min = out.grad.data / len(min_indices)
                for i in min_indices:
                    flat_grad[i] += grad_per_min

                self.grad.data = self._unflatten(flat_grad, self.shape)

        out._backward = _backward
        return out

    def dot(self, other: "Tensor") -> "Tensor":
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("Multiplicação matricial exige tensores 2D.")
        if self.shape[1] != other.shape[0]:
            raise ValueError("Dimensões incompatíveis para multiplicação matricial.")

        result_data = [
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.shape[1]))
                for j in range(other.shape[1])
            ]
            for i in range(self.shape[0])
        ]
        out = self._wrap_result(result_data, "dot", (self, other))

        def _backward():
            if self.requires_grad:
                grad_a = Tensor(out.grad.data).dot(other.transpose()).data
                self.grad.data = self._apply_recursive(
                    self.grad.data, grad_a, lambda a, b: a + b
                )
            if other.requires_grad:
                grad_b = self.transpose().dot(Tensor(out.grad.data)).data
                other.grad.data = self._apply_recursive(
                    other.grad.data, grad_b, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def transpose(self) -> "Tensor":
        if len(self.shape) != 2:
            raise ValueError("A transposição é suportada apenas para tensores 2D.")
        rows, cols = self.shape
        transposed_data = [[self.data[j][i] for j in range(rows)] for i in range(cols)]
        out = self._wrap_result(transposed_data, "transpose", (self,))

        def _backward():
            if self.requires_grad:
                grad_transposed = Tensor(out.grad.data).transpose().data
                self.grad.data = self._apply_recursive(
                    self.grad.data, grad_transposed, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        out_data = self._apply_recursive(self.data, None, lambda a: math.exp(a))
        out = self._wrap_result(out_data, "exp", (self,))

        def _backward():
            if self.requires_grad:
                grad_data = self._apply_recursive(
                    out.data, out.grad.data, lambda a, b: a * b
                )
                self.grad.data = self._apply_recursive(
                    self.grad.data, grad_data, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out_data = self._apply_recursive(self.data, None, lambda a: max(0, a))
        out = self._wrap_result(out_data, "ReLU", (self,))

        def _backward():
            if self.requires_grad:
                grad_data = self._apply_recursive(
                    self.data, out.grad.data, lambda a, b: b if a > 0 else 0
                )
                self.grad.data = self._apply_recursive(
                    self.grad.data, grad_data, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def backward(self):
        if self.shape != ():
            raise ValueError(
                "O gradiente pode ser calculado apenas para tensores escalares."
            )

        topo: List[Tensor] = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        self.grad = Tensor(1.0)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        zero_data = self._apply_recursive(self.data, None, lambda _: 0.0)
        self.grad = Tensor(zero_data)

    @staticmethod
    def _flatten(data: Union[list, Number]) -> list:
        if isinstance(data, list):
            return [item for sublist in data for item in Tensor._flatten(sublist)]
        return [data]

    @staticmethod
    def _unflatten(flat_list: list, original_shape: tuple) -> Union[list, float]:
        if not original_shape:
            return flat_list[0]
        sub_size = len(flat_list) // original_shape[0]
        return [
            Tensor._unflatten(
                flat_list[i * sub_size : (i + 1) * sub_size], original_shape[1:]
            )
            for i in range(original_shape[0])
        ]

    @staticmethod
    def concatenate(tensors: List["Tensor"], axis: int = 0) -> "Tensor":
        if not tensors:
            raise ValueError(
                "A lista de tensores para concatenar não pode estar vazia."
            )
        if axis == 0:
            combined_data = []
            for t in tensors:
                data = t.data if isinstance(t.data, list) else [t.data]
                combined_data.extend(data)
            return Tensor(
                combined_data, requires_grad=any(t.requires_grad for t in tensors)
            )
        elif axis == 1:
            if any(len(t.shape) < 2 for t in tensors):
                raise ValueError(
                    "Concatenação no eixo 1 exige que todos os tensores sejam pelo menos 2D."
                )
            num_rows = tensors[0].shape[0]
            if any(t.shape[0] != num_rows for t in tensors):
                raise ValueError(
                    "Todos os tensores devem ter o mesmo número de linhas para concatenar no eixo 1."
                )
            combined_data = [[] for _ in range(num_rows)]
            for t in tensors:
                for i, row in enumerate(t.data):
                    combined_data[i].extend(row if isinstance(row, list) else [row])
            return Tensor(
                combined_data, requires_grad=any(t.requires_grad for t in tensors)
            )
        else:
            raise NotImplementedError(
                "A concatenação só é suportada para axis=0 ou axis=1."
            )

    def __truediv__(self, other):
        return self * (other**-1)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (other * -1.0)

    def __rsub__(self, other):
        return other + (self * -1.0)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return other * (self**-1)
