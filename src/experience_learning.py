"""Evidence-gated learning signals from completed experiences."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import exp
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Experience:
    experience_id: str
    state: str
    action: str
    outcome: str
    reward: float
    success: bool
    cost: float = 0.0
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LearningSignal:
    experience_id: str
    reward: float
    advantage: float
    priority: float
    success: bool
    reason: str


@dataclass(frozen=True)
class CreditAssignment:
    experience_id: str
    credit: float
    rank: int


@dataclass(frozen=True)
class PolicyUpdate:
    state: str
    action: str
    value_delta: float
    count: int
    confidence: float


@dataclass(frozen=True)
class LearningReport:
    signals: tuple[LearningSignal, ...]
    credits: tuple[CreditAssignment, ...]
    updates: tuple[PolicyUpdate, ...]
    baseline: float
    fingerprint: str


def _fingerprint(parts: Sequence[str]) -> str:
    return sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


class RewardModel:
    def score(self, experience: Experience) -> float:
        base = float(experience.reward)
        base += 0.25 if experience.success else -0.25
        base -= max(0.0, float(experience.cost)) * 0.01
        return max(-1.0, min(1.0, base))


class AdvantageEstimator:
    def estimate(self, rewards: Sequence[float], gamma: float = 0.95) -> tuple[float, ...]:
        if not rewards:
            return ()
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        baseline = sum(rewards) / len(rewards)
        return tuple((reward - baseline) * (gamma ** index) for index, reward in enumerate(rewards))


class PriorityReplay:
    def __init__(self, epsilon: float = 0.05) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon

    def order(self, signals: Sequence[LearningSignal]) -> tuple[LearningSignal, ...]:
        return tuple(sorted(
            signals,
            key=lambda item: (-abs(item.advantage) - self.epsilon, item.experience_id),
        ))


class CreditAssigner:
    def assign(self, signals: Sequence[LearningSignal]) -> tuple[CreditAssignment, ...]:
        if not signals:
            return ()
        raw = [abs(signal.advantage) + 1e-6 for signal in signals]
        total = sum(raw)
        order = sorted(range(len(signals)), key=lambda index: (-raw[index], signals[index].experience_id))
        ranks = {index: rank + 1 for rank, index in enumerate(order)}
        return tuple(
            CreditAssignment(signals[index].experience_id, raw[index] / total, ranks[index])
            for index in range(len(signals))
        )


class TabularPolicyAdapter:
    def update(self, experiences: Sequence[Experience], signals: Sequence[LearningSignal]) -> tuple[PolicyUpdate, ...]:
        grouped: dict[tuple[str, str], list[float]] = {}
        for experience, signal in zip(experiences, signals):
            grouped.setdefault((experience.state, experience.action), []).append(signal.advantage)
        updates = []
        for (state, action), values in sorted(grouped.items()):
            delta = sum(values) / len(values)
            confidence = 1.0 - exp(-len(values) / 3.0)
            updates.append(PolicyUpdate(state, action, delta, len(values), confidence))
        return tuple(updates)


class ExperienceLearningEngine:
    def __init__(self, gamma: float = 0.95) -> None:
        self.reward_model = RewardModel()
        self.advantage = AdvantageEstimator()
        self.replay = PriorityReplay()
        self.credit = CreditAssigner()
        self.policy = TabularPolicyAdapter()
        self.gamma = gamma

    def learn(self, experiences: Sequence[Experience]) -> LearningReport:
        if not experiences:
            return LearningReport((), (), (), 0.0, _fingerprint(()))
        rewards = tuple(self.reward_model.score(item) for item in experiences)
        baseline = sum(rewards) / len(rewards)
        advantages = self.advantage.estimate(rewards, self.gamma)
        raw_signals = tuple(
            LearningSignal(
                item.experience_id,
                reward,
                advantage,
                abs(advantage) + 0.05,
                item.success,
                "verified outcome" if item.success else "failure feedback",
            )
            for item, reward, advantage in zip(experiences, rewards, advantages)
        )
        signals = self.replay.order(raw_signals)
        credits = self.credit.assign(signals)
        lookup = {item.experience_id: item for item in experiences}
        ordered_experiences = tuple(lookup[item.experience_id] for item in signals)
        updates = self.policy.update(ordered_experiences, signals)
        fingerprint = _fingerprint([f"{item.experience_id}:{item.advantage:.8f}" for item in signals])
        return LearningReport(signals, credits, updates, baseline, fingerprint)
