from typing import Any
from mytorch import Tensor

"This is an abstract class for layers."
class Layer:
    def __init__(self, need_bias: bool = False) -> None:
        self.need_bias = need_bias
        pass

    def __call__(self, inp: Tensor) -> Tensor:
        return self.forward(inp)
    
    def forward(self, x: Tensor) -> Tensor:
        return None
    
    def initialize(self):
        pass

    def zero_grad(self):
        pass

    def parameters(self):
        return None

    def __str__(self) -> str:
        return "Layer class is an abstract for other type of layers"
