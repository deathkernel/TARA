"""Weight initialization utilities for TARA.

Research basis:
- Glorot & Bengio (2010), Understanding the difficulty of training deep
  feedforward neural networks.
- He et al. (2015), Delving Deep into Rectifiers (used as a comparison for
  activation-specific initialization).

TARA uses Xavier/Glorot uniform initialization for its small tanh/GELU-style
linear projections. The goal is to keep activation scales reasonable at the
start of training without adding external dependencies.
"""

import math


def xavier_uniform_limit(fan_in, fan_out):
    """Return the symmetric Xavier/Glorot-uniform bound."""
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError("fan_in and fan_out must be positive")
    return math.sqrt(6.0 / (fan_in + fan_out))


def xavier_uniform(rng, fan_in, fan_out):
    """Draw one deterministic Xavier/Glorot-uniform sample from ``rng``."""
    limit = xavier_uniform_limit(fan_in, fan_out)
    return rng.uniform(-limit, limit)
