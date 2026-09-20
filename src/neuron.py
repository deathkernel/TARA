"""Single trainable neuron for TARA.

The neuron is intentionally built on TARA's own Value/autograd engine.
"""

from src.autograd import Value


class Neuron:
    def __init__(self, weight=0.0, bias=0.0):
        self.w = Value(weight)
        self.b = Value(bias)

    def forward(self, x):
        x = x if isinstance(x, Value) else Value(x)
        return (self.w * x + self.b).relu()

    def parameters(self):
        return [self.w, self.b]
