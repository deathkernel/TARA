"""Small decoder-style Transformer components for TARA.

Research reference: Vaswani et al. (2017), Attention Is All You Need.
This version adds LayerNorm, multi-head causal self-attention, residual
connections, and a GELU-style feed-forward network while keeping the model
small and fully compatible with TARA's scalar autodiff engine.
"""

import math
import random

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


class LayerNorm:
    """Layer normalization over one token vector."""

    def __init__(self, dimension, eps=1e-5):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.eps = eps
        self.gamma = [Value(1.0) for _ in range(dimension)]
        self.beta = [Value(0.0) for _ in range(dimension)]

    def forward(self, x):
        if len(x) != self.dimension:
            raise ValueError("input dimension must match LayerNorm dimension")
        mean = x[0]
        for value in x[1:]:
            mean = mean + value
        mean = mean / self.dimension

        centered = [value - mean for value in x]
        variance = centered[0] * centered[0]
        for value in centered[1:]:
            variance = variance + value * value
        variance = variance / self.dimension
        inv_std = (variance + self.eps) ** -0.5
        return [self.gamma[i] * centered[i] * inv_std + self.beta[i]
                for i in range(self.dimension)]

    def parameters(self):
        return self.gamma + self.beta


class CausalSelfAttentionHead:
    """One scaled dot-product causal attention head."""

    def __init__(self, embedding_dim, head_dim, rng):
        limit = 1.0 / math.sqrt(embedding_dim)
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim
        self.wq = [[Value(rng.uniform(-limit, limit)) for _ in range(embedding_dim)]
                   for _ in range(head_dim)]
        self.wk = [[Value(rng.uniform(-limit, limit)) for _ in range(embedding_dim)]
                   for _ in range(head_dim)]
        self.wv = [[Value(rng.uniform(-limit, limit)) for _ in range(embedding_dim)]
                   for _ in range(head_dim)]
        self.bq = [Value(0.0) for _ in range(head_dim)]
        self.bk = [Value(0.0) for _ in range(head_dim)]
        self.bv = [Value(0.0) for _ in range(head_dim)]

    @staticmethod
    def _project(token, matrix, bias):
        output = []
        for row, offset in zip(matrix, bias):
            value = offset
            for weight, item in zip(row, token):
                value = value + weight * item
            output.append(value)
        return output

    @staticmethod
    def _softmax(values):
        maximum = max(value.data for value in values)
        shifted = [(value - maximum).exp() for value in values]
        total = shifted[0]
        for value in shifted[1:]:
            total = total + value
        return [value / total for value in shifted]

    def forward(self, sequence):
        queries = [self._project(token, self.wq, self.bq) for token in sequence]
        keys = [self._project(token, self.wk, self.bk) for token in sequence]
        values = [self._project(token, self.wv, self.bv) for token in sequence]
        outputs = []
        scale = math.sqrt(self.head_dim)
        for i, query in enumerate(queries):
            scores = []
            for j in range(i + 1):
                score = query[0] * keys[j][0]
                for dimension in range(1, self.head_dim):
                    score = score + query[dimension] * keys[j][dimension]
                scores.append(score / scale)
            weights = self._softmax(scores)
            output = []
            for dimension in range(self.head_dim):
                value = weights[0] * values[0][dimension]
                for j in range(1, i + 1):
                    value = value + weights[j] * values[j][dimension]
                output.append(value)
            outputs.append(output)
        return outputs

    def parameters(self):
        return ([p for row in self.wq for p in row] + [p for row in self.wk for p in row] +
                [p for row in self.wv for p in row] + self.bq + self.bk + self.bv)


class MultiHeadCausalSelfAttention:
    """Small multi-head causal self-attention with a learned output projection."""

    def __init__(self, embedding_dim, num_heads, rng):
        if num_heads <= 0 or embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        head_dim = embedding_dim // num_heads
        self.heads = [CausalSelfAttentionHead(embedding_dim, head_dim, rng)
                      for _ in range(num_heads)]
        self.output = Linear(embedding_dim, embedding_dim, rng)

    def forward(self, sequence):
        per_head = [head.forward(sequence) for head in self.heads]
        combined = []
        for position in range(len(sequence)):
            token = []
            for head in per_head:
                token.extend(head[position])
            combined.append(self.output.forward(token))
        return combined

    def parameters(self):
        parameters = []
        for head in self.heads:
            parameters.extend(head.parameters())
        parameters.extend(self.output.parameters())
        return parameters


class TransformerBlock:
    """One small pre-norm decoder-style Transformer block."""

    def __init__(self, embedding_dim, ff_dim=None, num_heads=2, seed=0):
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        ff_dim = ff_dim or (embedding_dim * 2)
        rng = random.Random(seed)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.norm1 = LayerNorm(embedding_dim)
        self.attention = MultiHeadCausalSelfAttention(embedding_dim, num_heads, rng)
        self.norm2 = LayerNorm(embedding_dim)
        self.ff1 = Linear(embedding_dim, ff_dim, rng)
        self.ff2 = Linear(ff_dim, embedding_dim, rng)

    @staticmethod
    def _gelu(value):
        # Standard tanh approximation to GELU, expressed with Value ops.
        coefficient = math.sqrt(2.0 / math.pi)
        return 0.5 * value * (1.0 + (coefficient * (value + 0.044715 * value ** 3)).tanh())

    def forward(self, sequence):
        if not sequence:
            raise ValueError("sequence must not be empty")
        if any(len(token) != self.embedding_dim for token in sequence):
            raise ValueError("all token vectors must match embedding_dim")

        x = add_sinusoidal_position(sequence)
        normalized = [self.norm1.forward(token) for token in x]
        attended = self.attention.forward(normalized)
        after_attention = [[a + b for a, b in zip(residual, update)]
                           for residual, update in zip(x, attended)]

        output = []
        for residual in after_attention:
            normalized = self.norm2.forward(residual)
            hidden = [self._gelu(value) for value in self.ff1.forward(normalized)]
            update = self.ff2.forward(hidden)
            output.append([a + b for a, b in zip(residual, update)])
        return output

    def parameters(self):
        return (self.norm1.parameters() + self.attention.parameters() +
                self.norm2.parameters() + self.ff1.parameters() + self.ff2.parameters())

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0.0
