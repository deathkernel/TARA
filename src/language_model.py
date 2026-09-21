"""Tiny character-level autoregressive language model for TARA.

Research basis:
- Vaswani et al. (2017), Attention Is All You Need.
- GPT-style causal language modeling: predict the next token from the
  tokens at or before the current position.

This is deliberately tiny and educational: a small Transformer stack, one
character tokenizer, and scalar reverse-mode autodiff.
"""

import math
import random

from src.embeddings import Embedding
from src.transformer import Linear, TransformerStack


def logsumexp(values):
    """Numerically stable log(sum(exp(values))) for scalar Values."""
    if not values:
        raise ValueError("logsumexp requires at least one value")
    maximum = max(value.data for value in values)
    shifted = [(value - maximum).exp() for value in values]
    total = shifted[0]
    for value in shifted[1:]:
        total = total + value
    return total.log() + maximum


def cross_entropy(logits, target_id):
    """Stable negative log-likelihood for one target token."""
    if not logits:
        raise ValueError("logits must not be empty")
    if not 0 <= target_id < len(logits):
        raise IndexError("target_id out of vocabulary range")
    return logsumexp(logits) - logits[target_id]


class TinyLanguageModel:
    """Small autoregressive language model with a configurable Transformer stack."""

    def __init__(
        self,
        vocab_size,
        embedding_dim=8,
        ff_dim=16,
        seed=0,
        num_heads=None,
        num_layers=1,
    ):
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if num_heads is None:
            num_heads = 2 if embedding_dim % 2 == 0 else 1

        self.embedding = Embedding(
            vocab_size,
            embedding_dim=embedding_dim,
            seed=seed,
        )
        self.transformer = TransformerStack(
            embedding_dim,
            ff_dim=ff_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            seed=seed + 1,
        )
        self.lm_head = Linear(
            embedding_dim,
            vocab_size,
            random.Random(seed + 2),
        )

    def forward(self, token_ids):
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        vectors = [self.embedding.forward(token_id) for token_id in token_ids]
        hidden = self.transformer.forward(vectors)
        return [self.lm_head.forward(vector) for vector in hidden]

    def loss(self, inputs, targets):
        if not inputs or not targets:
            raise ValueError("inputs and targets must not be empty")
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets must have the same length")
        logits = self.forward(inputs)
        losses = [
            cross_entropy(row, target)
            for row, target in zip(logits, targets)
        ]
        total = losses[0]
        for value in losses[1:]:
            total = total + value
        return total / len(losses)

    def parameters(self):
        return (
            self.embedding.parameters()
            + self.transformer.parameters()
            + self.lm_head.parameters()
        )

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0.0

    def next_token(self, token_ids):
        """Return the highest-probability next token for a context."""
        logits = self.forward(token_ids)[-1]
        return max(range(len(logits)), key=lambda index: logits[index].data)

    def sample_next_token(self, token_ids, temperature=1.0, top_k=None, rng=None):
        """Sample a next token using temperature and optional top-k filtering."""
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")

        logits = [value.data for value in self.forward(token_ids)[-1]]
        scaled = [logit / temperature for logit in logits]

        if top_k is not None:
            top_k = min(top_k, len(scaled))
            keep = set(
                sorted(
                    range(len(scaled)),
                    key=scaled.__getitem__,
                    reverse=True,
                )[:top_k]
            )
            filtered = [
                value if index in keep else float("-inf")
                for index, value in enumerate(scaled)
            ]
        else:
            filtered = scaled

        maximum = max(filtered)
        weights = [
            math.exp(value - maximum) if math.isfinite(value) else 0.0
            for value in filtered
        ]
        total = sum(weights)
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("invalid sampling probability distribution")

        rng = random if rng is None else rng
        threshold = rng.random() * total
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += weight
            if threshold < cumulative:
                return index
        return len(weights) - 1
