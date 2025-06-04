import numpy as np
from mytorch import Tensor, Dependency

def tanh(x: Tensor) -> Tensor:
    """
    Implements the tanh function using Tensor.
    The tanh function is defined as (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
    """

    # Calculate tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    data = np.tanh(x.data)  # Using numpy's built-in tanh for forward calculation
    req_grad = x.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray):
            # Derivative of tanh: 1 - tanh(x)^2
            return grad * (1 - data ** 2)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
