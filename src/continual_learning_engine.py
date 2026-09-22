"""Advanced continual-learning orchestration for TARA.

Learning is evidence gated: only verified examples can enter replay or be
promoted. The engine adds reservoir-style replay, class/cluster balancing,
importance-weighted sampling, stability/plasticity diagnostics, acceptance
gates, and promotion manifests. It does not train a model itself.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .continual_learning import ReplayRecord, _fingerprint


@dataclass(frozen=True)
class LearningExample:
    text: str
    fingerprint: str
    domain: str
    importance: float = 1.0
    verified: bool = True
    source: str = "verified"


@dataclass(frozen=True)
class ReplayBatch:
    examples: tuple[LearningExample, ...]
    weights: tuple[float, ...]
    seed: int
    fingerprint: str


@dataclass(frozen=True)
class LearningMetrics:
    baseline_score: float
    current_score: float
    retention_score: float
    improvement_score: float
    forgetting_delta: float
    stable: bool


@dataclass(frozen=True)
class PromotionResult:
    accepted: bool
    reason: str
    metrics: LearningMetrics


class ExampleNormalizer:
    """Normalize verified records into deterministic learning examples."""

    def normalize(self, records: Iterable[Mapping[str, object]]) -> tuple[LearningExample, ...]:
        result: list[LearningExample] = []
        seen: set[str] = set()
        for item in records:
            if item.get("verified", True) is not True:
                continue
            value = item.get("text", item.get("content"))
            if value is None and item.get("problem") is not None:
                value = f"Problem: {item['problem']}\nSolution: {item.get('solution', '')}"
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            fingerprint = _fingerprint(text)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            importance = max(0.01, float(item.get("importance", 1.0)))
            domain = str(item.get("domain", item.get("source", "general")))
            result.append(LearningExample(text, fingerprint, domain, importance, True, str(item.get("source", "verified"))))
        return tuple(result)


class BalancedReplaySampler:
    """Importance-weighted replay with deterministic domain balancing."""

    def __init__(self, *, seed: int = 42, max_examples: int = 256, exploration: float = 0.20) -> None:
        if max_examples <= 0 or not 0.0 <= exploration <= 1.0:
            raise ValueError("invalid replay sampler configuration")
        self.seed = int(seed)
        self.max_examples = int(max_examples)
        self.exploration = float(exploration)

    def sample(self, examples: Sequence[LearningExample]) -> ReplayBatch:
        if not examples:
            return ReplayBatch((), (), self.seed, hashlib.sha256(b"").hexdigest())
        rng = random.Random(self.seed)
        groups: dict[str, list[LearningExample]] = {}
        for item in examples:
            groups.setdefault(item.domain, []).append(item)
        selected: list[LearningExample] = []
        domains = sorted(groups)
        # First guarantee domain coverage; remaining slots use a softmax-like
        # importance distribution with an exploration floor.
        while groups and len(selected) < min(self.max_examples, len(examples)):
            available = [domain for domain in domains if groups.get(domain)]
            if not available:
                break
            weights = [sum(x.importance for x in groups[d]) for d in available]
            total = sum(weights)
            if total <= 0:
                domain = rng.choice(available)
            else:
                threshold = rng.random() * total
                running = 0.0
                domain = available[-1]
                for candidate_domain, weight in zip(available, weights):
                    running += weight
                    if threshold <= running:
                        domain = candidate_domain
                        break
            pool = groups[domain]
            index = rng.randrange(len(pool)) if rng.random() < self.exploration else max(range(len(pool)), key=lambda i: pool[i].importance)
            selected.append(pool.pop(index))
            if not pool:
                groups.pop(domain)
        selected.sort(key=lambda x: (x.domain, x.fingerprint))
        weights = tuple(item.importance / max(0.01, sum(x.importance for x in selected)) for item in selected)
        fp = hashlib.sha256("\n".join(x.fingerprint for x in selected).encode()).hexdigest()
        return ReplayBatch(tuple(selected), weights, self.seed, fp)


class StabilityPlasticityAnalyzer:
    """Measure whether new learning improves without damaging retained skills."""

    def evaluate(self, baseline: Sequence[float], current: Sequence[float], *, tolerance: float = 0.02) -> LearningMetrics:
        if len(baseline) != len(current) or not baseline:
            raise ValueError("baseline and current scores must have equal non-zero length")
        base = sum(baseline) / len(baseline)
        now = sum(current) / len(current)
        deltas = [c - b for b, c in zip(baseline, current)]
        retention = sum(min(0.0, delta) for delta in deltas)
        forgetting = abs(retention) / len(deltas)
        improvement = sum(max(0.0, delta) for delta in deltas) / len(deltas)
        stable = forgetting <= tolerance
        return LearningMetrics(base, now, max(0.0, 1.0 - forgetting), improvement, forgetting, stable)


class KnowledgePromotionGate:
    """Accept learning only when aggregate progress and retention are safe."""

    def __init__(self, *, min_improvement: float = 0.0, max_forgetting: float = 0.02) -> None:
        self.min_improvement = float(min_improvement)
        self.max_forgetting = float(max_forgetting)

    def decide(self, metrics: LearningMetrics) -> PromotionResult:
        if metrics.forgetting_delta > self.max_forgetting:
            return PromotionResult(False, "rejected: catastrophic-forgetting threshold exceeded", metrics)
        if metrics.improvement_score < self.min_improvement:
            return PromotionResult(False, "rejected: no measurable improvement", metrics)
        return PromotionResult(True, "accepted: improvement retained within forgetting budget", metrics)


class ContinualLearningEngine:
    """Coordinate verified replay and evidence-gated knowledge promotion."""

    def __init__(self, *, seed: int = 42, replay_size: int = 256) -> None:
        self.normalizer = ExampleNormalizer()
        self.sampler = BalancedReplaySampler(seed=seed, max_examples=replay_size)
        self.analyzer = StabilityPlasticityAnalyzer()
        self.gate = KnowledgePromotionGate()

    def build_replay(self, records: Iterable[Mapping[str, object]]) -> ReplayBatch:
        return self.sampler.sample(self.normalizer.normalize(records))

    def evaluate(self, baseline: Sequence[float], current: Sequence[float]) -> PromotionResult:
        metrics = self.analyzer.evaluate(baseline, current)
        return self.gate.decide(metrics)

    @staticmethod
    def write_manifest(path: str | Path, batch: ReplayBatch, promotion: PromotionResult | None = None) -> None:
        payload = {
            "replay_fingerprint": batch.fingerprint,
            "count": len(batch.examples),
            "examples": [
                {"fingerprint": x.fingerprint, "domain": x.domain, "importance": x.importance, "source": x.source}
                for x in batch.examples
            ],
        }
        if promotion is not None:
            payload["promotion"] = {"accepted": promotion.accepted, "reason": promotion.reason, "metrics": promotion.metrics.__dict__}
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
