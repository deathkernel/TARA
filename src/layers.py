"""Basic dense layer implementation for TARA."""

from dataclasses import dataclass
import numpy as np


@dataclass
class DenseLayer:
    input_size: int
    output_size: int

    def __post_init__(self) -> None:
        limit = np.sqrt(2.0 / self.input_size)
        self.weights = np.random.randn(self.input_size, self.output_size) * limit
        self.bias = np.zeros((1, self.output_size))
        self.input_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        return x @ self.weights + self.bias

    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        if self.input_cache is None:
            raise RuntimeError("forward() must be called before backward().")

        grad_weights = self.input_cache.T @ grad_output
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = grad_output @ self.weights.T

        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        return grad_input
