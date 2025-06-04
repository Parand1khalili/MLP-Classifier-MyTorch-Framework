from mytorch.optimizer import Optimizer
from typing import List
from mytorch.layer import Layer
import numpy as np

class Adam(Optimizer):
    def __init__(self, layers: List[Layer], lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super().__init__(layers)
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.moments_first = {id(param): np.zeros_like(param.data) for layer in layers for param in layer.parameters()}
        self.moments_second = {id(param): np.zeros_like(param.data) for layer in layers for param in layer.parameters()}
        self.step_count = 0

    def step(self):
        """Executes a single optimization step using the Adam algorithm."""
        self.step_count += 1
        for layer in self.layers:
            for param in layer.parameters():
                if param.requires_grad:
                    # Update biased first moment (moving average of gradients)
                    first_moment = self.moments_first[id(param)]
                    first_moment = self.beta_1 * first_moment + (1 - self.beta_1) * param.grad.data
                    self.moments_first[id(param)] = first_moment

                    # Update biased second moment (moving average of squared gradients)
                    second_moment = self.moments_second[id(param)]
                    second_moment = self.beta_2 * second_moment + (1 - self.beta_2) * (param.grad.data ** 2)
                    self.moments_second[id(param)] = second_moment

                    # Bias-corrected moments
                    first_unbias = first_moment / (1 - self.beta_1 ** self.step_count)
                    second_unbias = second_moment / (1 - self.beta_2 ** self.step_count)

                    # Update parameter using Adam's rule
                    param.data -= self.lr * first_unbias / (np.sqrt(second_unbias) + self.epsilon)