"""Generation utilities for TARA language models.

Research basis:
- Temperature, top-k, and nucleus (top-p) filtering are standard decoding
  controls used in modern autoregressive language-model generation systems.

TARA keeps the implementation small and dependency-free. Greedy decoding,
temperature sampling, top-k filtering, and nucleus filtering are provided
without changing learned model parameters.
"""

import math
import random


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

    scaled = [float(logit) / temperature for logit in logits]
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
    """Generate text using configurable decoding controls."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt must not be empty")

    for _ in range(length):
        context = ids[-context_length:]
        logits = [_logit_value(value) for value in model.forward(context)[-1]]
        next_id = sample_from_logits(
            logits, temperature=temperature, top_k=top_k, top_p=top_p, rng=rng
        )
        ids.append(next_id)
    return tokenizer.decode(ids)
