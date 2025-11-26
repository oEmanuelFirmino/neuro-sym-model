from typing import List
from src.tensor.tensor import Tensor


class SGD:
    def __init__(self, parameters: List[Tensor], lr: float):
        if not isinstance(parameters, list):
            raise TypeError("Os parâmetros devem ser uma lista de Tensores.")
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = lr

    def step(self):
        for p in self.parameters:
            if p.grad is not None:
                update = Tensor(p.grad.data) * self.lr
                p.data = (Tensor(p.data) - update).data

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()
