from mytorch.optimizer import Optimizer
from typing import List
from mytorch.layer import Layer
import numpy as np

class RMSprop(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.sq_grad = {id(param): np.zeros_like(param.data) for layer in layers for param in layer.parameters()}

    def step(self):
        "TODO: implement RMSprop algorithm"
        for layer in self.layers:
            for param in layer.parameters():
                if param.requires_grad:
                    # Update squared gradient
                    sq_g = self.sq_grad[id(param)]
                    sq_g = self.beta * sq_g + (1 - self.beta) * (param.grad.data ** 2)
                    self.sq_grad[id(param)] = sq_g

                    # Update parameter with RMSprop update rule
                    param.data -= self.learning_rate * param.grad.data / (np.sqrt(sq_g) + self.epsilon)
