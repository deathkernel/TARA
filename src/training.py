"""Small, explicit training utilities for TARA."""

from src.losses import MSELoss, mean


class SGD:
    """Plain stochastic-gradient-descent parameter update."""

    def __init__(self, parameters, learning_rate=0.05):
        self.parameters = list(parameters)
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.learning_rate = float(learning_rate)

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = 0.0

    def step(self):
        for parameter in self.parameters:
            parameter.data -= self.learning_rate * parameter.grad


def mse_batch(model, inputs, targets):
    """Build one mean-MSE computation graph for a batch."""
    if len(inputs) != len(targets) or not inputs:
        raise ValueError("inputs and targets must be non-empty and have equal length")
    loss = MSELoss()
    return mean(loss(model.forward(x), y) for x, y in zip(inputs, targets))


def train_step(model, optimizer, inputs, targets):
    """Run one full forward -> backward -> update step and return loss."""
    optimizer.zero_grad()
    total_loss = mse_batch(model, inputs, targets)
    total_loss.backward()
    optimizer.step()
    return total_loss.data
