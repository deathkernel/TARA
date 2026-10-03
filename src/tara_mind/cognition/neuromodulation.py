"""Neuromodulatory control signals for TARA Baby.

Grounding: biological neuromodulatory systems alter learning and behavioral
state. Recent work continues to refine links between reward timing, dopamine
signals and adaptive learning rates. TARA implements an abstract controller that
modulates learning/replay priority from novelty, surprise and outcome value; it
is not a biochemical simulation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NeuromodulatoryState:
    novelty: float
    reward: float
    arousal: float
    learning_gain: float
    replay_gain: float


class NeuromodulatoryController:
    """Convert experience signals into bounded learning-control gains."""

    def __init__(self, base_gain: float = 1.0) -> None:
        if base_gain <= 0:
            raise ValueError("base_gain must be > 0")
        self.base_gain = float(base_gain)

    def update(self, novelty: float, reward: float, surprise: float) -> NeuromodulatoryState:
        for name, value in (("novelty", novelty), ("reward", reward)):
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")
        if surprise < 0:
            raise ValueError("surprise must be >= 0")
        novelty = float(novelty)
        reward = float(reward)
        surprise = float(surprise)
        arousal = max(0.0, min(1.0, 0.55 * abs(novelty) + 0.45 * min(1.0, surprise)))
        learning_gain = self.base_gain * (0.75 + 0.75 * arousal + 0.25 * max(0.0, reward))
        replay_gain = 1.0 + 0.8 * arousal + 0.4 * max(0.0, reward)
        return NeuromodulatoryState(novelty, reward, arousal, learning_gain, replay_gain)
