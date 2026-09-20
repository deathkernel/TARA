"""Small scalar-autograd self-attention for TARA.

Research reference: Vaswani et al. (2017), Attention Is All You Need.
TARA intentionally starts with one causal attention head so the mechanism
stays inspectable and runnable on a normal PC.
"""

import math

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

    Each token is represented by a fixed-size vector. Q, K and V projections
    are deliberately omitted at this first stage; attention operates directly
    on the supplied representations so the core algorithm is easy to inspect.
    """

    def __init__(self, embedding_dim):
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim

    def forward(self, sequence):
        if not sequence:
            raise ValueError("sequence must not be empty")
        if any(len(token) != self.embedding_dim for token in sequence):
            raise ValueError("all token vectors must match embedding_dim")

        outputs = []
        scale = math.sqrt(self.embedding_dim)
        for i, query in enumerate(sequence):
            scores = []
            for j in range(i + 1):
                key = sequence[j]
                score = query[0] * key[0]
                for dimension in range(1, self.embedding_dim):
                    score = score + query[dimension] * key[dimension]
                scores.append(score / scale)

            weights = _softmax(scores)
            output = []
            for dimension in range(self.embedding_dim):
                value = weights[0] * sequence[0][dimension]
                for j in range(1, i + 1):
                    value = value + weights[j] * sequence[j][dimension]
                output.append(value)
            outputs.append(output)
        return outputs
