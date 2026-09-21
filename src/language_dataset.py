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

    def _indices(self, shuffle=False, seed=0):
        indices = list(range(len(self)))
        if shuffle:
            random.Random(seed).shuffle(indices)
        return indices

    def batch_at(self, batch_index, batch_size, shuffle=False, seed=0):
        """Return one deterministic mini-batch without materializing all batches."""
        if not isinstance(batch_index, int) or isinstance(batch_index, bool):
            raise TypeError("batch_index must be an integer")
        if batch_index < 0:
            raise ValueError("batch_index must be non-negative")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        indices = self._indices(shuffle=shuffle, seed=seed)
        start = batch_index * batch_size
        selected = indices[start:start + batch_size]
        if not selected:
            raise IndexError("batch index out of range")
        return [self[index] for index in selected]

    def batch_count(self, batch_size):
        """Return the number of mini-batches for the current dataset size."""
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return (len(self) + batch_size - 1) // batch_size

    def iter_batches(self, batch_size, shuffle=False, seed=0):
        """Yield mini-batches of causal windows."""
        for batch_index in range(self.batch_count(batch_size)):
            yield self.batch_at(
                batch_index,
                batch_size,
                shuffle=shuffle,
                seed=seed,
            )


def dataset_statistics(dataset, unk_id=None):
    """Return transparent token/window statistics for an experiment."""
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
