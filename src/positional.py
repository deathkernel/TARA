"""Sinusoidal positional encoding for TARA.

Reference: Vaswani et al. (2017), Attention Is All You Need.
Fixed positional signals make token order visible without adding parameters.
"""

import math

from src.autograd import Value


def add_sinusoidal_position(sequence):
    """Add deterministic sinusoidal position signals to token vectors."""
    if not sequence:
        raise ValueError("sequence must not be empty")
    dimension = len(sequence[0])
    if dimension <= 0 or any(len(token) != dimension for token in sequence):
        raise ValueError("all token vectors must have the same positive dimension")

    result = []
    for position, token in enumerate(sequence):
        encoded = []
        for i, value in enumerate(token):
            angle = position / (10000 ** (2 * (i // 2) / dimension))
            signal = math.sin(angle) if i % 2 == 0 else math.cos(angle)
            encoded.append(value + signal)
        result.append(encoded)
    return result
