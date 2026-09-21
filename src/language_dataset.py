"""Causal language-model dataset utilities for TARA.

Research basis:
- Vaswani et al. (2017), Transformer sequence modeling.
- Autoregressive language modeling trains on next-token targets: a context
  of tokens predicts the token immediately following each position.

The implementation stays dependency-free and keeps the complete token stream
in memory so experiments remain easy to inspect on a normal PC.
"""


class CausalTextDataset:
    """Expose fixed-length causal next-token windows from token IDs."""

    def __init__(self, token_ids, context_length):
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        self.token_ids = list(token_ids)
        if len(self.token_ids) < 2:
            raise ValueError("token_ids must contain at least two tokens")
        self.context_length = context_length

    def __len__(self):
        return max(0, len(self.token_ids) - 1)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("dataset index out of range")
        start = index
        end = min(start + self.context_length + 1, len(self.token_ids))
        window = self.token_ids[start:end]
        return window[:-1], window[1:]

    def iter_windows(self):
        """Yield non-overlapping context windows for efficient full-corpus passes."""
        for start in range(0, len(self.token_ids) - 1, self.context_length):
            end = min(start + self.context_length + 1, len(self.token_ids))
            window = self.token_ids[start:end]
            if len(window) >= 2:
                yield window[:-1], window[1:]


def split_token_ids(token_ids, validation_fraction=0.1):
    """Split an ordered token stream without shuffling temporal order."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    token_ids = list(token_ids)
    if len(token_ids) < 4:
        raise ValueError("token_ids must contain at least four tokens")

    split_index = int(len(token_ids) * (1.0 - validation_fraction))
    split_index = min(max(split_index, 1), len(token_ids) - 1)
    train_ids = token_ids[:split_index]
    validation_ids = token_ids[split_index:]
    if len(train_ids) < 2 or len(validation_ids) < 2:
        raise ValueError("split must leave at least two tokens in each partition")
    return train_ids, validation_ids


def build_causal_datasets(tokenizer, text, context_length, validation_fraction=0.1):
    """Encode text once, split it deterministically, and build train/validation datasets."""
    if not hasattr(tokenizer, "encode"):
        raise TypeError("tokenizer must provide encode(text)")
    if not text:
        raise ValueError("text must not be empty")

    token_ids = tokenizer.encode(text)
    train_ids, validation_ids = split_token_ids(
        token_ids,
        validation_fraction=validation_fraction,
    )
    return (
        CausalTextDataset(train_ids, context_length),
        CausalTextDataset(validation_ids, context_length),
    )
