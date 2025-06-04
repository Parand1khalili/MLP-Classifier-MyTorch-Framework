from mytorch import Tensor, Dependency
import numpy as np

def MeanSquaredError(preds: Tensor, actual: Tensor) -> Tensor:
    """
    Implements Mean Squared Error loss.
    Args:
        preds (Tensor): Predicted values.
        actual (Tensor): Ground truth values.
    """
    # Calculate MSE: (1/n) * sum((y - y_hat)^2)
    data = np.mean((preds.data - actual.data) ** 2)

    if preds.requires_grad:
        def grad_fn(grad: np.ndarray):
            # Gradient of MSE: 2 * (preds - actual) / n
            return grad * 2 * (preds.data - actual.data) / preds.data.shape[0]

        depends_on = [Dependency(preds, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=preds.requires_grad, depends_on=depends_on)
