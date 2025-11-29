import random
import math
from abc import ABC, abstractmethod
from typing import Dict
from ..tensor import Tensor


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

    def state_dict(self) -> Dict[str, list]:
        state = {name: p.data for name, p in self._parameters.items()}
        for name, module in self._modules.items():
            state[name] = module.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict):
        for name, data in state_dict.items():
            if name in self._parameters:
                self._parameters[name].data = data
            elif name in self._modules:
                self._modules[name].load_state_dict(data)

    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        limit = math.sqrt(2.0 / in_features)
        weights_data = [
            [random.normalvariate(0, limit) for _ in range(out_features)]
            for _ in range(in_features)
        ]
        bias_data = [[0.0 for _ in range(out_features)]]
        self.weights = Tensor(weights_data, requires_grad=True)
        self.bias = Tensor(bias_data, requires_grad=True)
        self.add_parameter("weights", self.weights)
        self.add_parameter("bias", self.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x.dot(self.weights) + self.bias


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(1.0) / (Tensor(1.0) + (x * -1.0).exp())


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x
