"""Small embedding layer for TARA's language-learning path."""

import operator
import random

from src.autograd import Value


class Embedding:
    """Learn one dense vector for every token in a vocabulary."""

    def __init__(self, vocab_size, embedding_dim=8, seed=0):
        if vocab_size <= 0 or embedding_dim <= 0:
            raise ValueError("vocab_size and embedding_dim must be positive")
        rng = random.Random(seed)
        self.table = [
            [Value(rng.uniform(-0.1, 0.1)) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ]

    def forward(self, token_id):
        if isinstance(token_id, bool):
            raise TypeError("token_id must be an integer")
        try:
            token_id = operator.index(token_id)
        except TypeError as exc:
            raise TypeError("token_id must be an integer") from exc
        if not 0 <= token_id < len(self.table):
            raise IndexError("token_id out of vocabulary range")
        return self.table[token_id]

    def parameters(self):
        return [parameter for row in self.table for parameter in row]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0.0
