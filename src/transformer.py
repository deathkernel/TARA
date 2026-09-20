"""A tiny decoder-style Transformer block for TARA.

Research reference: Vaswani et al. (2017), Attention Is All You Need.
This block keeps one attention head and omits LayerNorm initially so every
operation remains easy to inspect in TARA's scalar autodiff engine.
"""

import random

from src.attention import SelfAttention
from src.autograd import Value
from src.positional import add_sinusoidal_position


class Linear:
    """Small fully connected vector projection using scalar Value objects."""

    def __init__(self, nin, nout, rng):
        if nin <= 0 or nout <= 0:
            raise ValueError("linear dimensions must be positive")
        limit = (1.0 / nin) ** 0.5
        self.w = [[Value(rng.uniform(-limit, limit)) for _ in range(nin)] for _ in range(nout)]
        self.b = [Value(0.0) for _ in range(nout)]

    def forward(self, x):
        output = []
        for row, bias in zip(self.w, self.b):
            value = bias
            for weight, item in zip(row, x):
                value = value + weight * item
            output.append(value)
        return output

    def parameters(self):
        return [p for row in self.w for p in row] + self.b


class TransformerBlock:
    """One small causal Transformer-style block."""

    def __init__(self, embedding_dim, ff_dim=None, seed=0):
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        ff_dim = ff_dim or (embedding_dim * 2)
        rng = random.Random(seed)
        self.embedding_dim = embedding_dim
        self.attention = SelfAttention(embedding_dim, seed=rng.randrange(2**32))
        self.attention_out = Linear(embedding_dim, embedding_dim, rng)
        self.ff1 = Linear(embedding_dim, ff_dim, rng)
        self.ff2 = Linear(ff_dim, embedding_dim, rng)

    def forward(self, sequence):
        if not sequence:
            raise ValueError("sequence must not be empty")
        if any(len(token) != self.embedding_dim for token in sequence):
            raise ValueError("all token vectors must match embedding_dim")

        x = add_sinusoidal_position(sequence)
        attended = self.attention.forward(x)
        after_attention = []
        for residual, update in zip(x, attended):
            projected = self.attention_out.forward(update)
            after_attention.append([a + b for a, b in zip(residual, projected)])

        output = []
        for residual in after_attention:
            hidden = [value.tanh() for value in self.ff1.forward(residual)]
            update = self.ff2.forward(hidden)
            output.append([a + b for a, b in zip(residual, update)])
        return output

    def parameters(self):
        return (self.attention.parameters() + self.attention_out.parameters() +
                self.ff1.parameters() + self.ff2.parameters())

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0.0
