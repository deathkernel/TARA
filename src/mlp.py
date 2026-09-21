"""Autograd-native multilayer perceptron for TARA.

Research references:
- Rumelhart, Hinton & Williams (1986), back-propagation.
- Glorot & Bengio (2010), Xavier/Glorot initialization.
- Karpathy's micrograd, used only as a conceptual reference.

TARA implements its own neural-network components on top of src.autograd.Value.
"""

import random

from src.autograd import Value
from src.initialization import xavier_uniform


class Neuron:
    """A fully connected neuron operating on scalar Value objects."""

    def __init__(self, nin, nonlin=True, rng=None):
        if nin <= 0:
            raise ValueError("nin must be positive")
        rng = rng or random.Random()
        self.w = [Value(xavier_uniform(rng, nin, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def forward(self, x):
        if len(x) != len(self.w):
            raise ValueError(f"expected {len(self.w)} inputs, got {len(x)}")
        values = [item if isinstance(item, Value) else Value(item) for item in x]
        activation = self.b
        for weight, value in zip(self.w, values):
            activation = activation + weight * value
        return activation.tanh() if self.nonlin else activation

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """A collection of independent fully connected neurons."""

    def __init__(self, nin, nout, nonlin=True, seed=None):
        if nout <= 0:
            raise ValueError("nout must be positive")
        rng = random.Random(seed)
        self.neurons = [Neuron(nin, nonlin=nonlin, rng=rng) for _ in range(nout)]

    def forward(self, x):
        outputs = [neuron.forward(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [parameter for neuron in self.neurons for parameter in neuron.parameters()]


class MLP:
    """A feed-forward multilayer perceptron.

    ``nouts`` describes each layer's number of neurons. Hidden layers use
    tanh; the final layer is linear so it can be used for regression/losses.
    """

    def __init__(self, nin, nouts, seed=None):
        if nin <= 0:
            raise ValueError("nin must be positive")
        if not nouts:
            raise ValueError("nouts must contain at least one layer")

        sizes = [nin] + list(nouts)
        rng = random.Random(seed)
        self.layers = []
        for index in range(len(nouts)):
            self.layers.append(
                Layer(
                    sizes[index],
                    sizes[index + 1],
                    nonlin=index != len(nouts) - 1,
                    seed=rng.randrange(2**32),
                )
            )

    def forward(self, x):
        values = [item if isinstance(item, Value) else Value(item) for item in x]
        for layer in self.layers:
            result = layer.forward(values)
            values = result if isinstance(result, list) else [result]
        return values[0] if len(values) == 1 else values

    def parameters(self):
        return [parameter for layer in self.layers for parameter in layer.parameters()]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0.0
