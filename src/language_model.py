"""Tiny character-level autoregressive language model for TARA.

Research basis:
- Vaswani et al. (2017), Attention Is All You Need.
- GPT-style causal language modeling: predict the next token from the
  tokens at or before the current position.

This is deliberately tiny and educational: one Transformer block, one
character tokenizer, and scalar reverse-mode autodiff.
"""

from src.autograd import Value
from src.embeddings import Embedding
from src.tokenizer import CharTokenizer
from src.transformer import Linear, TransformerBlock


def _softmax(values):
    maximum = max(value.data for value in values)
    shifted = [(value - maximum).exp() for value in values]
    total = shifted[0]
    for value in shifted[1:]:
        total = total + value
    return [value / total for value in shifted]


def cross_entropy(logits, target_id):
    """Negative log probability of one target token."""
    if not logits:
        raise ValueError("logits must not be empty")
    if not 0 <= target_id < len(logits):
        raise IndexError("target_id out of vocabulary range")
    probabilities = _softmax(logits)
    return -probabilities[target_id].log()


class TinyLanguageModel:
    """One-block character language model with causal next-token prediction."""

    def __init__(self, vocab_size, embedding_dim=8, ff_dim=16, seed=0):
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.embedding = Embedding(vocab_size, embedding_dim=embedding_dim, seed=seed)
        self.transformer = TransformerBlock(embedding_dim, ff_dim=ff_dim, seed=seed + 1)
        self.lm_head = Linear(embedding_dim, vocab_size, __import__('random').Random(seed + 2))

    def forward(self, token_ids):
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        vectors = [self.embedding.forward(token_id) for token_id in token_ids]
        hidden = self.transformer.forward(vectors)
        return [self.lm_head.forward(vector) for vector in hidden]

    def loss(self, inputs, targets):
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets must have the same length")
        logits = self.forward(inputs)
        losses = [cross_entropy(row, target) for row, target in zip(logits, targets)]
        total = losses[0]
        for value in losses[1:]:
            total = total + value
        return total / len(losses)

    def parameters(self):
        return (self.embedding.parameters() + self.transformer.parameters() +
                self.lm_head.parameters())

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0.0

    def next_token(self, token_ids):
        """Return the highest-probability next token for a context."""
        logits = self.forward(token_ids)[-1]
        return max(range(len(logits)), key=lambda index: logits[index].data)
