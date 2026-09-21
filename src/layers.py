"""Basic dense layer implementation for TARA."""

from dataclasses import dataclass
import numbers

import numpy as np


@dataclass
class DenseLayer:
    input_size: int
    output_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.input_size, numbers.Integral) or self.input_size <= 0:
            raise ValueError("input_size must be a positive integer")
        if not isinstance(self.output_size, numbers.Integral) or self.output_size <= 0:
            raise ValueError("output_size must be a positive integer")
        limit = np.sqrt(2.0 / self.input_size)
        self.weights = np.random.randn(self.input_size, self.output_size) * limit
        self.bias = np.zeros((1, self.output_size))
        self.input_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(
                f"expected a 2D input with shape (batch, {self.input_size}), got {x.shape}"
            )
        self.input_cache = x
        return x @ self.weights + self.bias

    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        if self.input_cache is None:
            raise RuntimeError("forward() must be called before backward().")
        if not isinstance(learning_rate, numbers.Real) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive number")
        if not isinstance(grad_output, np.ndarray):
            grad_output = np.asarray(grad_output)
        if grad_output.shape != (self.input_cache.shape[0], self.output_size):
            raise ValueError(
                "grad_output shape must match the layer output shape "
                f"({self.input_cache.shape[0]}, {self.output_size}), got {grad_output.shape}"
            )

        grad_weights = self.input_cache.T @ grad_output
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = grad_output @ self.weights.T

        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        return grad_input
