"""First-order optimizers for TARA.

Research basis:
- Kingma & Ba (2014), Adam: A Method for Stochastic Optimization.
- Loshchilov & Hutter (2017), Decoupled Weight Decay Regularization (AdamW).

AdamW is implemented here as a small, dependency-free scalar optimizer so
TARA can study adaptive optimization without hiding the mathematics inside a
framework. The optimizer keeps first and second moment estimates per scalar
parameter and applies decoupled weight decay.
"""

import math


class AdamW:
    """Adam with decoupled weight decay for scalar ``Value`` parameters."""

    def __init__(self, parameters, learning_rate=1e-3, betas=(0.9, 0.999), epsilon=1e-8, weight_decay=0.0):
        self.parameters = list(parameters)
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if len(betas) != 2 or not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas must contain two values in [0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        self.learning_rate = float(learning_rate)
        self.beta1, self.beta2 = map(float, betas)
        self.epsilon = float(epsilon)
        self.weight_decay = float(weight_decay)
        self.step_count = 0
        self._first_moment = [0.0] * len(self.parameters)
        self._second_moment = [0.0] * len(self.parameters)

    def step(self, learning_rate=None):
        """Apply one AdamW update and return the global gradient norm."""
        lr = self.learning_rate if learning_rate is None else float(learning_rate)
        if lr <= 0:
            raise ValueError("learning_rate must be positive")

        gradients = []
        squared_norm = 0.0
        for parameter in self.parameters:
            gradient = float(parameter.grad)
            if not math.isfinite(gradient):
                raise ValueError("gradient must be finite")
            gradients.append(gradient)
            squared_norm += gradient * gradient

        next_step = self.step_count + 1
        correction1 = 1.0 - self.beta1 ** next_step
        correction2 = 1.0 - self.beta2 ** next_step
        for index, (parameter, gradient) in enumerate(zip(self.parameters, gradients)):
            first = self.beta1 * self._first_moment[index] + (1.0 - self.beta1) * gradient
            second = self.beta2 * self._second_moment[index] + (1.0 - self.beta2) * gradient * gradient
            first_hat = first / correction1
            second_hat = second / correction2
            parameter.data *= 1.0 - lr * self.weight_decay
            parameter.data -= lr * first_hat / (math.sqrt(second_hat) + self.epsilon)
            self._first_moment[index] = first
            self._second_moment[index] = second
        self.step_count = next_step
        return math.sqrt(squared_norm)

    def state_dict(self):
        """Return JSON-compatible optimizer state for checkpointing."""
        return {
            "step_count": self.step_count,
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "first_moment": list(self._first_moment),
            "second_moment": list(self._second_moment),
        }

    def load_state_dict(self, state):
        """Restore optimizer state after validating its structure."""
        required = {
            "step_count", "learning_rate", "beta1", "beta2", "epsilon",
            "weight_decay", "first_moment", "second_moment",
        }
        if not isinstance(state, dict) or not required.issubset(state):
            raise ValueError("optimizer state is incomplete")
        first = state["first_moment"]
        second = state["second_moment"]
        if not isinstance(first, list) or not isinstance(second, list):
            raise ValueError("optimizer state is missing moment arrays")
        if len(first) != len(self.parameters) or len(second) != len(self.parameters):
            raise ValueError("optimizer state parameter count does not match")
        step_count = int(state["step_count"])
        learning_rate = float(state["learning_rate"])
        beta1 = float(state["beta1"])
        beta2 = float(state["beta2"])
        epsilon = float(state["epsilon"])
        weight_decay = float(state["weight_decay"])
        if step_count < 0 or learning_rate <= 0 or not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("optimizer state contains invalid configuration")
        if epsilon <= 0 or weight_decay < 0:
            raise ValueError("optimizer state contains invalid configuration")
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in second):
            raise ValueError("optimizer state contains invalid second moments")
        if any(not math.isfinite(float(value)) for value in first):
            raise ValueError("optimizer state contains non-finite first moments")
        self.step_count = step_count
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self._first_moment = [float(value) for value in first]
        self._second_moment = [float(value) for value in second]
