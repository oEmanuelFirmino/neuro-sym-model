from typing import Any, Tuple, Union, Iterable, Optional, List
from src.neurosym.tensor.backend import get_backend

Number = Union[int, float]


class Tensor:
    def __init__(
        self,
        data: Union[list, Number],
        requires_grad: bool = False,
        _op: Optional[str] = None,
        _parents: Optional[Iterable["Tensor"]] = None,
    ):
        self.backend = get_backend()
        if isinstance(data, (int, float)):
            self.data, self.shape = float(data), ()
        elif isinstance(data, list):
            self.data, self.shape = data, self._get_shape(data)
        else:
            raise TypeError(
                "Tensor aceita apenas listas aninhadas ou números (int/float)."
            )
        self.requires_grad = bool(requires_grad)
        self.grad: Optional[Tensor] = None
        if self.requires_grad:
            self.zero_grad()
        self._op, self._parents, self._backward = (
            _op,
            tuple(_parents) if _parents else tuple(),
            lambda: None,
        )

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
        return (
            (len(data),) + self._get_shape(data[0])
            if isinstance(data, list) and data
            else ()
        )

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}, op={self._op})"

    def _wrap_result(self, data, op_name, parents):
        requires_grad = any(p.requires_grad for p in parents)
        return Tensor(data, requires_grad=requires_grad, _op=op_name, _parents=parents)

    def _unbroadcast(self, grad_data: Any, target_shape: Tuple[int, ...]) -> Any:
        """Reduz o gradiente (soma) para corresponder ao shape original (unbroadcasting)."""
        current_shape = self._get_shape(grad_data)

        if current_shape == target_shape:
            return grad_data

        if not target_shape:
            return self.backend.sum(grad_data)

        if len(current_shape) > len(target_shape):

            if isinstance(grad_data, list) and len(grad_data) > 0:
                accumulated = grad_data[0]
                for i in range(1, len(grad_data)):
                    accumulated = self.backend.apply_recursive(
                        accumulated, grad_data[i], lambda a, b: a + b
                    )

                return self._unbroadcast(accumulated, target_shape)

        return grad_data

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.backend.apply_recursive(
            self.data, other.data, lambda a, b: a + b
        )
        out = self._wrap_result(out_data, "add", (self, other))

        def _backward():
            if self.requires_grad:
                grad_self = out.grad.data
                if self.shape != out.shape:
                    grad_self = self._unbroadcast(grad_self, self.shape)

                self.grad.data = self.backend.apply_recursive(
                    self.grad.data, grad_self, lambda a, b: a + b
                )

            if other.requires_grad:
                grad_other = out.grad.data
                if other.shape != out.shape:
                    grad_other = self._unbroadcast(grad_other, other.shape)

                other.grad.data = self.backend.apply_recursive(
                    other.grad.data, grad_other, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.backend.apply_recursive(
            self.data, other.data, lambda a, b: a * b
        )
        out = self._wrap_result(out_data, "mul", (self, other))

        def _backward():
            if self.requires_grad:
                term = self.backend.apply_recursive(
                    other.data, out.grad.data, lambda a, b: a * b
                )
                if self.shape != out.shape:
                    term = self._unbroadcast(term, self.shape)

                self.grad.data = self.backend.apply_recursive(
                    self.grad.data,
                    term,
                    lambda a, b: a + b,
                )

            if other.requires_grad:
                term = self.backend.apply_recursive(
                    self.data, out.grad.data, lambda a, b: a * b
                )
                if other.shape != out.shape:
                    term = self._unbroadcast(term, other.shape)

                other.grad.data = self.backend.apply_recursive(
                    other.grad.data,
                    term,
                    lambda a, b: a + b,
                )

        out._backward = _backward
        return out

    def __pow__(self, p: float):
        out_data = self.backend.apply_recursive(self.data, None, lambda a: a**p)
        out = self._wrap_result(out_data, f"pow({p})", (self,))

        def _backward():
            if self.requires_grad:
                grad_data = self.backend.apply_recursive(
                    self.data, out.grad.data, lambda a, b: (p * (a ** (p - 1))) * b
                )
                self.grad.data = self.backend.apply_recursive(
                    self.grad.data, grad_data, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def sum(self):
        out_data = self.backend.sum(self.data)
        out = self._wrap_result(out_data, "sum", (self,))

        def _backward():
            if self.requires_grad:
                self.grad.data = self.backend.apply_recursive(
                    self.grad.data, out.grad.data, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def mean(self):
        out_data = self.backend.mean(self.data)
        out = self._wrap_result(out_data, "mean", (self,))

        def _backward():
            if self.requires_grad:
                n = len(self._flatten(self.data))
                self.grad.data = self.backend.apply_recursive(
                    self.grad.data, out.grad.data / n, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def p_mean(self, p: float):
        out_data = self.backend.p_mean(self.data, p)
        out = self._wrap_result(out_data, f"p_mean({p})", (self,))

        def _backward():
            if self.requires_grad:
                n = len(self._flatten(self.data))
                pre_factor = (out.data ** (1 - p)) / n if out.data > 1e-9 else 0.0
                grad_data = self.backend.apply_recursive(
                    self.data,
                    out.grad.data,
                    lambda a, b: ((a + 1e-9) ** (p - 1)) * pre_factor * b,
                )
                self.grad.data = self.backend.apply_recursive(
                    self.grad.data, grad_data, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def min(self):
        out_data = self.backend.min(self.data)
        out = self._wrap_result(out_data, "min", (self,))

        def _backward():
            if self.requires_grad:
                flat_data, flat_grad = self._flatten(self.data), self._flatten(
                    self.grad.data
                )
                min_indices = [i for i, x in enumerate(flat_data) if x == out_data]
                grad_per_min = out.grad.data / len(min_indices)
                for i in min_indices:
                    flat_grad[i] += grad_per_min
                self.grad.data = self._unflatten(flat_grad, self.shape)

        out._backward = _backward
        return out

    def dot(self, other):
        out_data = self.backend.dot(self.data, other.data)
        out = self._wrap_result(out_data, "dot", (self, other))

        def _backward():
            if self.requires_grad:
                self.grad.data = self.backend.apply_recursive(
                    self.grad.data,
                    Tensor(out.grad.data).dot(other.transpose()).data,
                    lambda a, b: a + b,
                )
            if other.requires_grad:
                other.grad.data = self.backend.apply_recursive(
                    other.grad.data,
                    self.transpose().dot(Tensor(out.grad.data)).data,
                    lambda a, b: a + b,
                )

        out._backward = _backward
        return out

    def transpose(self):
        out_data = self.backend.transpose(self.data)
        out = self._wrap_result(out_data, "transpose", (self,))

        def _backward():
            if self.requires_grad:
                self.grad.data = self.backend.apply_recursive(
                    self.grad.data,
                    Tensor(out.grad.data).transpose().data,
                    lambda a, b: a + b,
                )

        out._backward = _backward
        return out

    def exp(self):
        out_data = self.backend.exp(self.data)
        out = self._wrap_result(out_data, "exp", (self,))

        def _backward():
            if self.requires_grad:
                self.grad.data = self.backend.apply_recursive(
                    self.grad.data,
                    self.backend.apply_recursive(
                        out.data, out.grad.data, lambda a, b: a * b
                    ),
                    lambda a, b: a + b,
                )

        out._backward = _backward
        return out

    def relu(self):
        out_data = self.backend.relu(self.data)
        out = self._wrap_result(out_data, "ReLU", (self,))

        def _backward():
            if self.requires_grad:

                grad_data = self.backend.relu_backward(out.grad.data, self.data)
                self.grad.data = self.backend.apply_recursive(
                    self.grad.data, grad_data, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def backward(self):
        if self.shape != ():
            raise ValueError(
                "O gradiente pode ser calculado apenas para tensores escalares."
            )
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                [build_topo(p) for p in v._parents]
                topo.append(v)

        build_topo(self)
        self.grad = Tensor(1.0)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        zero_data = self.backend.apply_recursive(self.data, None, lambda _: 0.0)
        self.grad = Tensor(zero_data)

    @staticmethod
    def _flatten(data: Union[list, Number]) -> list:
        return (
            [i for s in data for i in Tensor._flatten(s)]
            if isinstance(data, list)
            else [data]
        )

    @staticmethod
    def _unflatten(flat: list, shape: tuple) -> Union[list, float]:
        if not shape:
            return flat[0]
        size = len(flat) // shape[0]
        return [
            Tensor._unflatten(flat[i * size : (i + 1) * size], shape[1:])
            for i in range(shape[0])
        ]

    @staticmethod
    def concatenate(tensors: List["Tensor"], axis: int = 0) -> "Tensor":
        if not tensors:
            raise ValueError(
                "A lista de tensores para concatenar não pode estar vazia."
            )

        if axis == 0:
            data = []
            for t in tensors:
                if isinstance(t.data, list):
                    data.extend(t.data)
                else:
                    data.append(t.data)
        elif axis == 1:
            rows = tensors[0].shape[0]
            data = [[] for _ in range(rows)]
            for t in tensors:
                if t.shape[0] != rows:
                    raise ValueError(
                        "Todos os tensores devem ter o mesmo número de linhas para axis=1"
                    )
                for i, r in enumerate(t.data):
                    row_data = r if isinstance(r, list) else [r]
                    data[i].extend(row_data)
        else:
            raise NotImplementedError(
                "A concatenação só é suportada para axis=0 ou axis=1."
            )

        requires_grad = any(t.requires_grad for t in tensors)
        out = Tensor(
            data,
            requires_grad=requires_grad,
            _op="concatenate",
            _parents=tuple(tensors),
        )

        def _backward():
            if not out.grad:
                return

            current_idx = 0

            if axis == 0:
                for t in tensors:
                    dim_size = t.shape[0] if t.shape else 1

                    if t.requires_grad:
                        grad_slice = out.grad.data[current_idx : current_idx + dim_size]

                        if t.shape == ():
                            grad_slice = (
                                grad_slice[0]
                                if isinstance(grad_slice, list)
                                else grad_slice
                            )

                        if t.grad is None:
                            t.grad = Tensor(grad_slice)
                        else:
                            t.grad.data = t.backend.apply_recursive(
                                t.grad.data, grad_slice, lambda a, b: a + b
                            )

                    current_idx += dim_size

            elif axis == 1:
                for t in tensors:
                    cols = t.shape[1] if len(t.shape) > 1 else 1

                    if t.requires_grad:
                        grad_slice = []
                        for row in out.grad.data:
                            grad_slice.append(row[current_idx : current_idx + cols])

                        if t.grad is None:
                            t.grad = Tensor(grad_slice)
                        else:
                            t.grad.data = t.backend.apply_recursive(
                                t.grad.data, grad_slice, lambda a, b: a + b
                            )

                    current_idx += cols

        out._backward = _backward
        return out

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
