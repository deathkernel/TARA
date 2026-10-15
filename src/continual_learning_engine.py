"""Disabled compatibility types for the removed continual-learning pipeline.

TARA no longer collects replay examples, trains candidates, or promotes learned
model artifacts. These types keep older runtime interfaces importable while
ensuring no examples or model state are retained.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LearningExample:
    text: str = ""
    fingerprint: str = ""
    domain: str = ""
    importance: float = 1.0
    verified: bool = False
    source: str = "disabled"


@dataclass(frozen=True)
class ReplayBatch:
    examples: tuple[LearningExample, ...] = ()
    weights: tuple[float, ...] = ()
    seed: int = 0
    fingerprint: str = ""


@dataclass(frozen=True)
class LearningMetrics:
    baseline_score: float = 0.0
    current_score: float = 0.0
    retention_score: float = 0.0
    improvement_score: float = 0.0
    forgetting_delta: float = 0.0
    stable: bool = False


@dataclass(frozen=True)
class PromotionResult:
    accepted: bool
    reason: str
    metrics: LearningMetrics = LearningMetrics()


class ContinualLearningEngine:
    """No-op replacement: replay and candidate promotion are disabled."""

    def __init__(self, *args, **kwargs):
        self.seed = int(kwargs.get("seed", 0))

    def build_replay(self, records=()):
        return ReplayBatch(seed=self.seed)

    def evaluate(self, baseline, current):
        return PromotionResult(False, "continual learning and model promotion have been removed")
