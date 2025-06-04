from mytorch.optimizer import Optimizer
from typing import List
from mytorch.layer import Layer
import numpy as np

class Momentum(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate=0.01, momentum=0.9):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {id(param): np.zeros_like(param.data) for layer in layers for param in layer.parameters()}

    def step(self):
        "TODO: implement Momentum algorithm"
        for layer in self.layers:
            for param in layer.parameters():
                if param.requires_grad:
                    # Compute velocity update
                    v = self.velocity[id(param)]
                    v = self.momentum * v - self.learning_rate * param.grad.data
                    self.velocity[id(param)] = v

                    # Update parameter using velocity
                    param.data += v
