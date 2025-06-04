import numpy as np
from mytorch import Tensor, Dependency

def CategoricalCrossEntropyLoss(predictions: Tensor, targets: Tensor) -> Tensor:
    """
    Computes the Categorical Cross-Entropy loss.
    
    Args:
        predictions (Tensor): Model's output probabilities (softmaxed).
        targets (Tensor): Actual labels in one-hot encoded format.
    """
    # Safeguard against log(0) by clipping predictions within a stable range
    stable_preds = np.clip(predictions.data, 1e-12, 1 - 1e-12)
    
    # Compute cross-entropy loss: -mean(sum(target * log(prediction)))
    loss_value = -np.sum(targets.data * np.log(stable_preds)) / predictions.data.shape[0]

    # Define gradient function if gradient is required
    if predictions.requires_grad:
        def backward_fn(grad_output: np.ndarray):
            # Derivative of cross-entropy with respect to predictions
            return grad_output * (-targets.data / stable_preds) / predictions.data.shape[0]

        dependencies = [Dependency(predictions, backward_fn)]
    else:
        dependencies = []

    # Return computed loss as a Tensor with appropriate dependencies
    return Tensor(data=loss_value, requires_grad=predictions.requires_grad, depends_on=dependencies)
