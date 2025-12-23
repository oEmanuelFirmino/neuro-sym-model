from typing import Any, Tuple, Union, Iterable, Optional, List
from src.neurosym.tensor.backend import get_backend

Number = Union[int, float]


class Tensor:
    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        _op: Optional[str] = None,
        _parents: Optional[Iterable["Tensor"]] = None,
    ):
        self.backend = get_backend()

        # O backend cuida da conversão de List -> Array/Internal
        self.data = self.backend.convert_to_internal(data)

        # Shape delegado ao backend
        self.shape = self.backend.get_shape(self.data)

        self.requires_grad = bool(requires_grad)
        self.grad: Optional[Tensor] = None

        if self.requires_grad:
            self.zero_grad()

        self._op, self._parents, self._backward = (
            _op,
            tuple(_parents) if _parents else tuple(),
            lambda: None,
        )

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}, op={self._op})"

    def _wrap_result(self, data, op_name, parents):
        requires_grad = any(p.requires_grad for p in parents)
        return Tensor(data, requires_grad=requires_grad, _op=op_name, _parents=parents)

    # --- Operações Aritméticas ---

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.backend.apply_elementwise(
            self.data, other.data, lambda a, b: a + b
        )
        out = self._wrap_result(out_data, "add", (self, other))

        def _backward():
            if self.requires_grad:
                grad_self = out.grad.data
                grad_self = self.backend.unbroadcast(grad_self, self.shape)
                self.grad.data = self.backend.apply_elementwise(
                    self.grad.data, grad_self, lambda a, b: a + b
                )

            if other.requires_grad:
                grad_other = out.grad.data
                grad_other = self.backend.unbroadcast(grad_other, other.shape)
                other.grad.data = self.backend.apply_elementwise(
                    other.grad.data, grad_other, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.backend.apply_elementwise(
            self.data, other.data, lambda a, b: a * b
        )
        out = self._wrap_result(out_data, "mul", (self, other))

        def _backward():
            if self.requires_grad:
                term = self.backend.apply_elementwise(
                    other.data, out.grad.data, lambda a, b: a * b
                )
                term = self.backend.unbroadcast(term, self.shape)
                self.grad.data = self.backend.apply_elementwise(
                    self.grad.data, term, lambda a, b: a + b
                )

            if other.requires_grad:
                term = self.backend.apply_elementwise(
                    self.data, out.grad.data, lambda a, b: a * b
                )
                term = self.backend.unbroadcast(term, other.shape)
                other.grad.data = self.backend.apply_elementwise(
                    other.grad.data, term, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def __pow__(self, p: float):
        out_data = self.backend.apply_elementwise(self.data, None, lambda a: a**p)
        out = self._wrap_result(out_data, f"pow({p})", (self,))

        def _backward():
            if self.requires_grad:
                grad_data = self.backend.apply_elementwise(
                    self.data, out.grad.data, lambda a, b: (p * (a ** (p - 1))) * b
                )
                self.grad.data = self.backend.apply_elementwise(
                    self.grad.data, grad_data, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    # --- Reduções e Funções de Ativação ---

    def sum(self):
        out_data = self.backend.sum(self.data)
        out = self._wrap_result(out_data, "sum", (self,))

        def _backward():
            if self.requires_grad:
                grad_val = out.grad.data
                grad_tensor = self.backend.apply_elementwise(
                    self.data, grad_val, lambda a, b: b + (a * 0)
                )
                self.grad.data = self.backend.apply_elementwise(
                    self.grad.data, grad_tensor, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def mean(self):
        out_data = self.backend.mean(self.data)
        out = self._wrap_result(out_data, "mean", (self,))

        def _backward():
            if self.requires_grad:
                shape = self.shape
                n = 1
                for dim in shape:
                    n *= dim

                grad_per_element = out.grad.data / n if n > 0 else 0.0

                grad_tensor = self.backend.apply_elementwise(
                    self.data, grad_per_element, lambda a, b: b + (a * 0)
                )
                self.grad.data = self.backend.apply_elementwise(
                    self.grad.data, grad_tensor, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def p_mean(self, p: float):
        out_data = self.backend.p_mean(self.data, p)
        out = self._wrap_result(out_data, f"p_mean({p})", (self,))

        def _backward():
            if self.requires_grad:
                shape = self.shape
                n = 1
                for dim in shape:
                    n *= dim

                y_val = out.data
                pre_factor = (y_val ** (1 - p)) / n if n > 0 and y_val > 1e-9 else 0.0

                grad_calc = (
                    lambda x, g_out: ((x + 1e-9) ** (p - 1)) * pre_factor * g_out
                )

                grad_tensor = self.backend.apply_elementwise(
                    self.data, out.grad.data, grad_calc
                )
                self.grad.data = self.backend.apply_elementwise(
                    self.grad.data, grad_tensor, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def min(self):
        out_data = self.backend.min(self.data)
        out = self._wrap_result(out_data, "min", (self,))

        def _backward():
            if self.requires_grad:
                min_val = out.data
                grad_val = out.grad.data

                # Mascaramento booleano para gradiente do min (compatível com NumPy)
                grad_calc = lambda x, g: (abs(x - min_val) < 1e-9) * g

                grad_tensor = self.backend.apply_elementwise(
                    self.data, grad_val, grad_calc
                )
                self.grad.data = self.backend.apply_elementwise(
                    self.grad.data, grad_tensor, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def dot(self, other):
        out_data = self.backend.dot(self.data, other.data)
        out = self._wrap_result(out_data, "dot", (self, other))

        def _backward():
            if self.requires_grad:
                grad_contribution = self.backend.dot(
                    out.grad.data, self.backend.transpose(other.data)
                )
                self.grad.data = self.backend.apply_elementwise(
                    self.grad.data, grad_contribution, lambda a, b: a + b
                )

            if other.requires_grad:
                grad_contribution = self.backend.dot(
                    self.backend.transpose(self.data), out.grad.data
                )
                other.grad.data = self.backend.apply_elementwise(
                    other.grad.data, grad_contribution, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def transpose(self):
        out_data = self.backend.transpose(self.data)
        out = self._wrap_result(out_data, "transpose", (self,))

        def _backward():
            if self.requires_grad:
                grad_transposed = self.backend.transpose(out.grad.data)
                self.grad.data = self.backend.apply_elementwise(
                    self.grad.data, grad_transposed, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def exp(self):
        out_data = self.backend.exp(self.data)
        out = self._wrap_result(out_data, "exp", (self,))

        def _backward():
            if self.requires_grad:
                grad_calc = lambda y, g: y * g
                grad_tensor = self.backend.apply_elementwise(
                    out.data, out.grad.data, grad_calc
                )

                self.grad.data = self.backend.apply_elementwise(
                    self.grad.data,
                    grad_tensor,
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
                self.grad.data = self.backend.apply_elementwise(
                    self.grad.data, grad_data, lambda a, b: a + b
                )

        out._backward = _backward
        return out

    def backward(self):
        is_scalar = self.shape == () or (
            hasattr(self.data, "size") and self.data.size == 1
        )
        if not is_scalar:
            if (
                isinstance(self.data, list)
                and len(self.data) == 1
                and not isinstance(self.data[0], list)
            ):
                pass
            else:
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
        zero_data = self.backend.apply_elementwise(self.data, None, lambda x: 0.0)
        self.grad = Tensor(zero_data)

    @staticmethod
    def _flatten(data: Union[list, Number]) -> list:
        return (
            [i for s in data for i in Tensor._flatten(s)]
            if isinstance(data, list)
            else [data]
        )

    @staticmethod
    def concatenate(tensors: List["Tensor"], axis: int = 0) -> "Tensor":
        if not tensors:
            raise ValueError(
                "A lista de tensores para concatenar não pode estar vazia."
            )

        raw_data = [t.data for t in tensors]
        backend = get_backend()

        # Suporte ao Backend NumPy com correção para escalares (0-d arrays)
        if hasattr(backend, "convert_to_internal") and "NumpyBackend" in str(
            type(backend)
        ):
            import numpy as np

            try:
                # CORREÇÃO: Detecta arrays 0-d (escalares) e usa stack para empilhá-los em um vetor
                if (
                    raw_data
                    and isinstance(raw_data[0], np.ndarray)
                    and raw_data[0].ndim == 0
                ):
                    concat_data = np.stack(raw_data, axis=axis)
                else:
                    concat_data = np.concatenate(raw_data, axis=axis)
            except Exception as e:
                raise ValueError(f"Erro na concatenação NumPy: {e}")
        else:
            # Lógica manual para listas (PythonBackend)
            if axis == 0:
                concat_data = []
                for d in raw_data:
                    if isinstance(d, list):
                        concat_data.extend(d)
                    else:
                        concat_data.append(d)
            elif axis == 1:
                rows = len(raw_data[0])
                concat_data = [[] for _ in range(rows)]
                for d in raw_data:
                    for i, row in enumerate(d):
                        concat_data[i].extend(row if isinstance(row, list) else [row])
            else:
                raise NotImplementedError("Eixo não suportado para PythonBackend")

        requires_grad = any(t.requires_grad for t in tensors)
        out = Tensor(
            concat_data,
            requires_grad=requires_grad,
            _op="concatenate",
            _parents=tuple(tensors),
        )

        def _backward():
            if not out.grad:
                return
            pass

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
