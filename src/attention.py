"""Small scalar-autograd self-attention for TARA.

Research reference: Vaswani et al. (2017), Attention Is All You Need.
TARA starts with one causal attention head so the mechanism stays
inspectable and runnable on a normal PC.
"""

import math
import random

from src.autograd import Value


def _softmax(values):
    """Numerically stable softmax over scalar Value objects."""
    if not values:
        raise ValueError("softmax requires at least one value")
    maximum = max(value.data for value in values)
    shifted = [(value - maximum).exp() for value in values]
    total = shifted[0]
    for value in shifted[1:]:
        total = total + value
    return [value / total for value in shifted]


class SelfAttention:
    """Single-head scaled dot-product causal self-attention.

    The input is projected into learned query, key and value vectors. Future
    positions are masked by only scoring keys at or before the current token.
    """

    def __init__(self, embedding_dim, seed=0):
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        rng = random.Random(seed)
        limit = 1.0 / math.sqrt(embedding_dim)
        self.wq = self._matrix(rng, limit)
        self.wk = self._matrix(rng, limit)
        self.wv = self._matrix(rng, limit)
        self.bq = [Value(0.0) for _ in range(embedding_dim)]
        self.bk = [Value(0.0) for _ in range(embedding_dim)]
        self.bv = [Value(0.0) for _ in range(embedding_dim)]

    def _matrix(self, rng, limit):
        return [[Value(rng.uniform(-limit, limit)) for _ in range(self.embedding_dim)]
                for _ in range(self.embedding_dim)]

    def _project(self, token, matrix, bias):
        output = []
        for row, offset in zip(matrix, bias):
            value = offset
            for weight, item in zip(row, token):
                value = value + weight * item
            output.append(value)
        return output

    def forward(self, sequence):
        if not sequence:
            raise ValueError("sequence must not be empty")
        if any(len(token) != self.embedding_dim for token in sequence):
            raise ValueError("all token vectors must match embedding_dim")

        queries = [self._project(token, self.wq, self.bq) for token in sequence]
        keys = [self._project(token, self.wk, self.bk) for token in sequence]
        values = [self._project(token, self.wv, self.bv) for token in sequence]

        outputs = []
        scale = math.sqrt(self.embedding_dim)
        for i, query in enumerate(queries):
            scores = []
            for j in range(i + 1):
                score = query[0] * keys[j][0]
                for dimension in range(1, self.embedding_dim):
                    score = score + query[dimension] * keys[j][dimension]
                scores.append(score / scale)

            weights = _softmax(scores)
            output = []
            for dimension in range(self.embedding_dim):
                value = weights[0] * values[0][dimension]
                for j in range(1, i + 1):
                    value = value + weights[j] * values[j][dimension]
                output.append(value)
            outputs.append(output)
        return outputs

    def parameters(self):
        return ([parameter for row in self.wq for parameter in row] +
                [parameter for row in self.wk for parameter in row] +
                [parameter for row in self.wv for parameter in row] +
                self.bq + self.bk + self.bv)

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0.0
