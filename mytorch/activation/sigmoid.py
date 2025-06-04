import numpy as np
from mytorch import Tensor, Dependency

def sigmoid(x: Tensor) -> Tensor:
    """
    Implements the sigmoid function.
    Sigmoid is defined as 1 / (1 + exp(-x)).
    """
    # Sigmoid function: 1 / (1 + exp(-x))
    data = 1 / (1 + np.exp(-x.data))
    req_grad = x.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray):
            # Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
            sigmoid_data = 1 / (1 + np.exp(-x.data))
            return grad * sigmoid_data * (1 - sigmoid_data)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
