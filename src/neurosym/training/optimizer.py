from typing import List, Tuple
from src.neurosym.tensor.tensor import Tensor


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


class AdamW:
    """AdamW com weight decay desacoplado (Loshchilov & Hutter, 2019).

    Protocolo usado no artigo (Seção 5.4.1): lr=1e-3, weight_decay=1e-2.
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if not isinstance(parameters, list):
            raise TypeError("Os parâmetros devem ser uma lista de Tensores.")
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self._m = {id(p): [0.0] * len(Tensor._flatten(p.data)) for p in self.parameters}
        self._v = {id(p): [0.0] * len(Tensor._flatten(p.data)) for p in self.parameters}

    def step(self):
        self.t += 1
        bias_correction1 = 1 - self.beta1**self.t
        bias_correction2 = 1 - self.beta2**self.t

        for p in self.parameters:
            if p.grad is None:
                continue

            flat_param = Tensor._flatten(p.data)
            flat_grad = Tensor._flatten(p.grad.data)
            m, v = self._m[id(p)], self._v[id(p)]
            new_param = []

            for i, (theta, g) in enumerate(zip(flat_param, flat_grad)):
                m[i] = self.beta1 * m[i] + (1 - self.beta1) * g
                v[i] = self.beta2 * v[i] + (1 - self.beta2) * (g * g)
                m_hat = m[i] / bias_correction1
                v_hat = v[i] / bias_correction2
                theta = theta - self.lr * (
                    m_hat / (v_hat**0.5 + self.eps) + self.weight_decay * theta
                )
                new_param.append(theta)

            p.data = Tensor._unflatten(new_param, p.shape)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()
