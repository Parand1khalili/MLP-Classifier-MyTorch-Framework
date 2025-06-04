from typing import List
from mytorch.layer import Layer

"This is an abstract class for other optimizers"
class Optimizer:
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        pass

    def step(self):
        pass

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
