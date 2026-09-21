"""Causal language-model dataset utilities for TARA.

Research basis:
- Vaswani et al. (2017), Transformer sequence modeling.
- Autoregressive language modeling trains on next-token targets: a context
  of tokens predicts the token immediately following each position.

The implementation stays dependency-free and keeps the complete token stream
in memory so experiments remain easy to inspect on a normal PC.
"""

import random
from collections import Counter


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
        """Yield non-overlapping context windows for full-corpus passes."""
        for start in range(0, len(self.token_ids) - 1, self.context_length):
            end = min(start + self.context_length + 1, len(self.token_ids))
            window = self.token_ids[start:end]
            if len(window) >= 2:
                yield window[:-1], window[1:]

    def iter_batches(self, batch_size, shuffle=False, seed=0):
        """Yield mini-batches of causal windows.

        Every item is an independent ``(inputs, targets)`` pair. Windows are
        padded neither here nor in the model, so variable-length tail examples
        remain explicit and no artificial training tokens are introduced.
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        indices = list(range(len(self)))
        if shuffle:
            random.Random(seed).shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch = [self[index] for index in indices[start:start + batch_size]]
            if batch:
                yield batch


def dataset_statistics(dataset, unk_id=None):
    """Return transparent token/window statistics for an experiment.

    ``unk_id`` is optional so callers can measure unknown-token rate when the
    tokenizer exposes an explicit UNK token. The calculation is deterministic
    and does not inspect model outputs.
    """
    token_count = len(dataset.token_ids)
    window_count = len(dataset)
    unique_tokens = len(set(dataset.token_ids))
    unknown_tokens = (
        sum(token == unk_id for token in dataset.token_ids)
        if unk_id is not None
        else None
    )
    return {
        "token_count": token_count,
        "unique_tokens": unique_tokens,
        "window_count": window_count,
        "context_length": dataset.context_length,
        "unknown_tokens": unknown_tokens,
        "unknown_rate": (
            unknown_tokens / token_count if unknown_tokens is not None and token_count else None
        ),
    }


def token_frequency(token_ids, limit=10):
    """Return the most frequent token IDs for corpus inspection."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    return Counter(token_ids).most_common(limit)


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
