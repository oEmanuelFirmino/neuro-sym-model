from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple


class TensorBackend(ABC):
    """
    Interface abstrata para backends de tensores.
    Define as operações fundamentais que devem ser implementadas
    por qualquer backend numérico (NumPy, Python puro, Torch, etc).
    """

    @abstractmethod
    def convert_to_internal(self, data: Any) -> Any:
        """
        Converte dados brutos (listas, números) para o formato nativo do backend.
        Deve ser chamado na inicialização do Tensor.
        """
        pass

    @abstractmethod
    def get_shape(self, data: Any) -> Tuple[int, ...]:
        """
        Retorna o shape dos dados nativos em O(1).
        """
        pass

    @abstractmethod
    def apply_elementwise(self, a: Any, b: Any, op: Callable) -> Any:
        """
        Aplica uma operação element-wise (ponto a ponto).
        O backend deve lidar com broadcasting se necessário.
        """
        pass

    @abstractmethod
    def unbroadcast(self, data: Any, target_shape: Tuple[int, ...]) -> Any:
        """
        Reduz o tensor (soma) para corresponder ao target_shape.
        Essencial para o cálculo correto de gradientes (backward pass)
        quando houve broadcasting no forward pass.
        """
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
