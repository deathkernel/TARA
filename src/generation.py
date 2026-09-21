"""Generation utilities for TARA language models.

Research basis:
- Temperature, top-k, and nucleus (top-p) filtering are standard decoding
  controls used in modern autoregressive language-model generation systems.

TARA keeps the implementation small. Decoding reads numeric logits without
constructing training-time autodiff graphs, then applies standard sampling
controls without changing learned model parameters.
"""

import math
import random

from src.numeric_inference import forward_numeric


def _validate_controls(temperature, top_k, top_p):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, rng=None):
    """Select one token from raw logits using standard sampling controls."""
    if not logits:
        raise ValueError("logits must not be empty")
    _validate_controls(temperature, top_k, top_p)

    scaled = [float(logit.data if hasattr(logit, "data") else logit) / temperature for logit in logits]
    candidates = list(range(len(scaled)))

    if top_k is not None:
        k = min(top_k, len(candidates))
        candidates.sort(key=scaled.__getitem__, reverse=True)
        candidates = candidates[:k]

    if top_p is not None and top_p < 1.0:
        candidates.sort(key=scaled.__getitem__, reverse=True)
        maximum = max(scaled[index] for index in candidates)
        weights = [math.exp(scaled[index] - maximum) for index in candidates]
        total = sum(weights)
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("invalid sampling probability distribution")
        probabilities = [weight / total for weight in weights]
        kept = []
        cumulative = 0.0
        for index, probability in zip(candidates, probabilities):
            kept.append(index)
            cumulative += probability
            if cumulative >= top_p:
                break
        candidates = kept

    maximum = max(scaled[index] for index in candidates)
    weights = [math.exp(scaled[index] - maximum) for index in candidates]
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("invalid sampling probability distribution")

    rng = random if rng is None else rng
    threshold = rng.random() * total
    cumulative = 0.0
    for index, weight in zip(candidates, weights):
        cumulative += weight
        if threshold < cumulative:
            return index
    return candidates[-1]


def _logit_value(value):
    """Return a numeric logit from either TARA Values or plain numbers."""
    return float(value.data) if hasattr(value, "data") else float(value)


def generate(model, tokenizer, prompt, length=40, context_length=12,
             temperature=1.0, top_k=None, top_p=None, rng=None):
    """Generate text using graph-free model inference and decoding controls."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt must not be empty")

    for _ in range(length):
        context = ids[-context_length:]
        logits = forward_numeric(model, context)[-1]
        next_id = sample_from_logits(
            logits, temperature=temperature, top_k=top_k, top_p=top_p, rng=rng
        )
        ids.append(next_id)
    return tokenizer.decode(ids)
