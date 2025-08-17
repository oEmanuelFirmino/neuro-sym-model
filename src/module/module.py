import random
import math
from abc import ABC, abstractmethod
from src.tensor.tensor import Tensor


class Module(ABC):
    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def add_module(self, name: str, module):
        if not isinstance(module, Module):
            raise TypeError(f"{module} is not a Module subclass")
        self._modules[name] = module
        setattr(self, name, module)

    def add_parameter(self, name: str, param: Tensor):
        if not isinstance(param, Tensor):
            raise TypeError(f"{param} is not a Tensor")
        self._parameters[name] = param
        setattr(self, name, param)

    def parameters(self) -> list[Tensor]:
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        limit = 1 / math.sqrt(in_features)
        weights_data = [
            [random.uniform(-limit, limit) for _ in range(out_features)]
            for _ in range(in_features)
        ]
        bias_data = [[random.uniform(-limit, limit) for _ in range(out_features)]]
        self.weights = Tensor(weights_data, requires_grad=True)
        self.bias = Tensor(bias_data, requires_grad=True)
        self.add_parameter("weights", self.weights)
        self.add_parameter("bias", self.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x.dot(self.weights) + self.bias


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return (x * -1.0).exp().__add__(1.0) ** -1.0


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
