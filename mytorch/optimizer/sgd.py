from typing import List
from mytorch.layer import Layer
from mytorch.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        "TODO: implement SGD algorithm"
        # Iterate over each layer's parameters (weights and biases)
        for layer in self.layers:
            for param in layer.parameters():
                if param.requires_grad:
                    # Update parameter with simple gradient descent rule
                    param.data -= self.learning_rate * param.grad.data
