import numpy as np
from mytorch import Tensor, Dependency

def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    #Leaky ReLU: x if x > 0, otherwise alpha * x
    data = np.where(x.data > 0, x.data, alpha * x.data)
    req_grad = x.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray):
            # The gradient of Leaky ReLU is 1 for x > 0, and alpha for x <= 0
            return grad * np.where(x.data > 0, 1, alpha)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
