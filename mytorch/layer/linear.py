from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer

import numpy as np


class Linear(Layer):
    def __init__(self, inputs: int, outputs: int, need_bias: bool = False, mode="xavier") -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        # Perform the forward pass (matrix multiplication with optional bias addition)
        result = x @ self.weight
        if self.need_bias:
            result += self.bias
        return result

    def initialize(self):
        # Initialize weights using the specified mode
        self.weight = Tensor(
            data=initializer((self.inputs, self.outputs), mode=self.initialize_mode),  # Initialize weight matrix
            requires_grad=True  # Enable gradient calculation
        )

        # Initialize bias to zero if need_bias is True
        if self.need_bias:
            self.bias = Tensor(
                data=initializer((1, self.outputs), mode="zero"),  # Bias initialized to zeros
                requires_grad=True  # Enable gradient calculation for bias
            )

    def zero_grad(self):
        # Reset gradients of weights and bias
        if self.weight.requires_grad:
            self.weight.zero_grad()
        if self.need_bias and self.bias.requires_grad:
            self.bias.zero_grad()

    def parameters(self):
        # Return list of trainable parameters (weight and optional bias)
        return [self.weight, self.bias] if self.need_bias else [self.weight]

    def __str__(self) -> str:
        return "linear - total param: {} - in: {}, out: {}".format(self.inputs * self.outputs, self.inputs, self.outputs)
