from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple, Union


class TensorBackend(ABC):
    @abstractmethod
    def convert_to_internal(self, data: Any) -> Any:
        """Converte dados brutos (listas/números) para o formato nativo do backend."""
        pass

    @abstractmethod
    def get_shape(self, data: Any) -> Tuple[int, ...]:
        """Retorna o shape dos dados nativos em O(1)."""
        pass

    @abstractmethod
    def apply_elementwise(self, a: Any, b: Any, op: Callable) -> Any:
        """Aplica uma operação element-wise (com broadcasting implícito)."""
        pass

    @abstractmethod
    def unbroadcast(self, data: Any, target_shape: Tuple[int, ...]) -> Any:
        """Reduz o tensor para o target_shape (usado no backward pass)."""
        pass

    # ... métodos de redução existentes (sum, mean, etc) permanecem ...
    @abstractmethod
    def sum(self, data: Any) -> Any:
        pass

    @abstractmethod
    def mean(self, data: Any) -> Any:
        pass

    @abstractmethod
    def min(self, data: Any) -> Any:
        pass

    @abstractmethod
    def p_mean(self, data: Any, p: float) -> Any:
        pass

    @abstractmethod
    def exp(self, data: Any) -> Any:
        pass

    @abstractmethod
    def relu(self, data: Any) -> Any:
        pass

    @abstractmethod
    def relu_backward(self, grad_data: Any, input_data: Any) -> Any:
        pass

    @abstractmethod
    def dot(self, a: Any, b: Any) -> Any:
        pass

    @abstractmethod
    def transpose(self, data: Any) -> Any:
        pass
