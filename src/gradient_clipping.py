"""Gradient-norm clipping utilities for TARA.

Research reference:
- Pascanu, Mikolov & Bengio (2013), "On the difficulty of training
  recurrent neural networks". Gradient norm clipping is proposed as a
  practical defense against exploding gradients.

TARA uses global L2-norm clipping across its scalar Value parameters.
"""

import math


def clip_grad_norm_(parameters, max_norm):
    """Clip gradients in-place to a global L2 norm.

    Returns the total gradient norm before clipping.
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")

    parameters = list(parameters)
    total_squared = 0.0
    for parameter in parameters:
        gradient = float(parameter.grad)
        if not math.isfinite(gradient):
            raise ValueError("gradient must be finite")
        total_squared += gradient * gradient

    total_norm = math.sqrt(total_squared)
    if total_norm > max_norm and total_norm > 0.0:
        scale = max_norm / total_norm
        for parameter in parameters:
            parameter.grad *= scale
    return total_norm
