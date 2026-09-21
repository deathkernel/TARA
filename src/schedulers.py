"""Learning-rate schedules for TARA.

Research basis: cosine annealing is a widely studied learning-rate schedule.
TARA uses a small deterministic implementation so training dynamics remain
inspectable on a normal PC.
"""

import math


class CosineAnnealing:
    """Cosine decay from an initial learning rate to a minimum learning rate."""

    def __init__(self, initial_lr, total_steps, min_lr=0.0):
        if initial_lr <= 0:
            raise ValueError("initial_lr must be positive")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if min_lr < 0 or min_lr > initial_lr:
            raise ValueError("min_lr must be in [0, initial_lr]")
        self.initial_lr = float(initial_lr)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)

    def get_lr(self, step):
        if step < 0:
            raise ValueError("step must be non-negative")
        progress = min(step, self.total_steps) / self.total_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine
