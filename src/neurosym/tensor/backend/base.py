from abc import ABC, abstractmethod
from typing import Any, Callable


class TensorBackend(ABC):

    @abstractmethod
    def apply_recursive(self, a: Any, b: Any, op: Callable) -> Any:
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

    @abstractmethod
    def sum(self, data: Any) -> float:
        pass

    @abstractmethod
    def mean(self, data: Any) -> float:
        pass

    @abstractmethod
    def p_mean(self, data: Any, p: float) -> float:
        pass

    @abstractmethod
    def min(self, data: Any) -> float:
        pass
