import numpy as np
from mytorch import Tensor, Dependency

def softmax(input_tensor: Tensor) -> Tensor:
    """Computes softmax function for input in a batched format."""
    
    # Normalize by subtracting max value to stabilize large exponentials, then exponentiate
    stable_exp = np.exp(input_tensor.data - np.max(input_tensor.data, axis=-1, keepdims=True))
    exp_sum = np.sum(stable_exp, axis=-1, keepdims=True)
    softmax_output = stable_exp / exp_sum  # Result of softmax

    # Check if gradient computation is needed
    requires_grad = input_tensor.requires_grad

    if requires_grad:
        def backward(grad_output: np.ndarray):
            # Calculate the gradient matrix for each input in the batch
            softmax_probs = softmax_output
            grad_matrix = softmax_probs * (grad_output - np.sum(grad_output * softmax_probs, axis=-1, keepdims=True))
            return grad_matrix

        dependencies = [Dependency(input_tensor, backward)]
    else:
        dependencies = []

    # Return Tensor with computed softmax data and dependencies
    return Tensor(data=softmax_output, requires_grad=requires_grad, depends_on=dependencies)
