"""Architecture-level self-optimization for TARA.

This layer optimizes the composition/configuration of cognitive components,
not source code blindly. Every candidate is evaluated by an injected evaluator
and must pass a deterministic safety/regression gate before promotion.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import math
from typing import Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ArchitectureComponent:
    name: str
    enabled: bool = True
    parameters: tuple[tuple[str, float], ...] = ()
    dependencies: tuple[str, ...] = ()
    critical: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("component name must not be empty")
        if any(not key.strip() for key, _ in self.parameters):
            raise ValueError("parameter names must not be empty")


@dataclass(frozen=True)
class ArchitectureVariant:
    variant_id: str
    components: tuple[ArchitectureComponent, ...]
    rationale: str = ""
    parent_id: str | None = None


@dataclass(frozen=True)
class ArchitectureMetrics:
    task_score: float
    robustness: float
    latency_ms: float
    memory_mb: float
    throughput: float
    failures: int = 0

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (
            self.task_score, self.robustness, self.latency_ms,
            self.memory_mb, self.throughput,
        )):
            raise ValueError("architecture metrics must be finite")
        if self.latency_ms < 0 or self.memory_mb < 0 or self.throughput < 0 or self.failures < 0:
            raise ValueError("resource metrics must be non-negative")


@dataclass(frozen=True)
class ArchitectureEvaluation:
    variant: ArchitectureVariant
    metrics: ArchitectureMetrics
    utility: float
    fingerprint: str


@dataclass(frozen=True)
class BottleneckReport:
    primary: str
    severity: float
    components: tuple[str, ...]
    recommendations: tuple[str, ...]


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reason: str
    baseline_utility: float
    candidate_utility: float
    selected_variant_id: str | None = None


@dataclass(frozen=True)
class OptimizationRound:
    baseline: ArchitectureEvaluation
    candidates: tuple[ArchitectureEvaluation, ...]
    bottleneck: BottleneckReport
    decision: PromotionDecision


@dataclass(frozen=True)
class OptimizationResult:
    final: ArchitectureEvaluation
    rounds: tuple[OptimizationRound, ...]
    history: tuple[ArchitectureEvaluation, ...]


def _stable_int(value: str) -> int:
    return int.from_bytes(sha256(value.encode("utf-8")).digest()[:4], "big")


def fingerprint_variant(variant: ArchitectureVariant) -> str:
    parts = [variant.variant_id]
    for component in sorted(variant.components, key=lambda item: item.name):
        parts.append(component.name)
        parts.append(str(component.enabled))
        parts.append(str(component.critical))
        parts.extend(f"{key}={value!r}" for key, value in component.parameters)
        parts.extend(component.dependencies)
    return sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


class ArchitectureUtility:
    """Normalize measurable quality/resource metrics into one optimization signal."""

    def score(
        self,
        metrics: ArchitectureMetrics,
        *,
        baseline: ArchitectureMetrics | None = None,
    ) -> float:
        quality = 0.60 * max(0.0, min(1.0, metrics.task_score))
        robustness = 0.20 * max(0.0, min(1.0, metrics.robustness))
        if baseline is None:
            speed = 1.0 / (1.0 + math.log1p(metrics.latency_ms))
            memory = 1.0 / (1.0 + math.log1p(metrics.memory_mb))
        else:
            speed = baseline.latency_ms / max(metrics.latency_ms, 1e-9)
            speed = max(0.0, min(1.25, speed)) / 1.25
            memory = baseline.memory_mb / max(metrics.memory_mb, 1e-9)
            memory = max(0.0, min(1.25, memory)) / 1.25
        throughput = math.tanh(max(0.0, metrics.throughput)) * 0.10
        failure_penalty = min(0.20, metrics.failures * 0.02)
        return quality + robustness + 0.05 * speed + 0.05 * memory + throughput - failure_penalty


class BottleneckDetector:
    """Use relative resource/quality pressure to identify the dominant constraint."""

    def diagnose(
        self,
        baseline: ArchitectureMetrics,
        *,
        latency_budget_ms: float | None = None,
        memory_budget_mb: float | None = None,
    ) -> BottleneckReport:
        pressures = {
            "quality": max(0.0, 1.0 - max(0.0, min(1.0, baseline.task_score))),
            "robustness": max(0.0, 1.0 - max(0.0, min(1.0, baseline.robustness))),
            "latency": (
                max(0.0, baseline.latency_ms / latency_budget_ms - 1.0)
                if latency_budget_ms and latency_budget_ms > 0 else 0.0
            ),
            "memory": (
                max(0.0, baseline.memory_mb / memory_budget_mb - 1.0)
                if memory_budget_mb and memory_budget_mb > 0 else 0.0
            ),
            "failures": min(1.0, baseline.failures / 10.0),
        }
        primary, raw = max(pressures.items(), key=lambda item: (item[1], item[0]))
        severity = max(0.0, min(1.0, raw if raw > 0 else 0.1))
        component_names = ()
        recommendations = {
            "quality": ("improve reasoning accuracy", "increase verification coverage"),
            "robustness": ("add fallback/retry paths", "increase input diversity"),
            "latency": ("disable expensive non-critical stages", "reduce repeated context computation"),
            "memory": ("reduce retained context", "compact high-cost representations"),
            "failures": ("harden error boundaries", "add regression cases"),
        }[primary]
        return BottleneckReport(primary, severity, component_names, recommendations)


class ArchitectureMutator:
    """Generate small, explainable architectural mutations."""

    def mutate(
        self,
        baseline: ArchitectureVariant,
        bottleneck: BottleneckReport,
        *,
        count: int = 4,
    ) -> tuple[ArchitectureVariant, ...]:
        if count <= 0:
            raise ValueError("count must be positive")
        components = list(baseline.components)
        candidates: list[ArchitectureVariant] = []

        def add_variant(label: str, changed: Sequence[ArchitectureComponent]) -> None:
            variant_id = f"{baseline.variant_id}:{label}:{len(candidates)}"
            candidates.append(
                ArchitectureVariant(
                    variant_id=variant_id,
                    components=tuple(changed),
                    rationale=label + "; " + "; ".join(bottleneck.recommendations),
                    parent_id=baseline.variant_id,
                )
            )

        if bottleneck.primary == "latency":
            for index, component in enumerate(components):
                if not component.critical and component.enabled:
                    changed = components.copy()
                    changed[index] = replace(component, enabled=False)
                    add_variant(f"disable-{component.name}", changed)
        elif bottleneck.primary == "memory":
            for index, component in enumerate(components):
                if component.parameters:
                    changed = components.copy()
                    params = tuple((key, value * 0.75) for key, value in component.parameters)
                    changed[index] = replace(component, parameters=params)
                    add_variant(f"compact-{component.name}", changed)
        elif bottleneck.primary in {"quality", "robustness"}:
            for index, component in enumerate(components):
                params = dict(component.parameters)
                changed = components.copy()
                params["verification_depth"] = params.get("verification_depth", 1.0) + 0.25
                changed[index] = replace(component, parameters=tuple(sorted(params.items())))
                add_variant(f"verify-{component.name}", changed)
        else:
            for index, component in enumerate(components):
                if not component.critical:
                    changed = components.copy()
                    changed[index] = replace(component, enabled=not component.enabled)
                    add_variant(f"toggle-{component.name}", changed)

        seen = set()
        unique = []
        for candidate in candidates:
            fp = fingerprint_variant(candidate)
            if fp not in seen and fp != fingerprint_variant(baseline):
                seen.add(fp)
                unique.append(candidate)
            if len(unique) >= count:
                break
        return tuple(unique)


class ArchitecturePromotionGate:
    """Require no material quality/robustness regression before promotion."""

    def __init__(self, *, quality_tolerance: float = 0.01, robustness_tolerance: float = 0.02):
        if quality_tolerance < 0 or robustness_tolerance < 0:
            raise ValueError("tolerances must be non-negative")
        self.quality_tolerance = quality_tolerance
        self.robustness_tolerance = robustness_tolerance

    def accept(self, baseline: ArchitectureEvaluation, candidate: ArchitectureEvaluation) -> bool:
        base, current = baseline.metrics, candidate.metrics
        return (
            current.task_score >= base.task_score - self.quality_tolerance
            and current.robustness >= base.robustness - self.robustness_tolerance
            and candidate.utility > baseline.utility
        )


class ArchitectureOptimizer:
    """Bounded architecture search with measured promotion."""

    def __init__(
        self,
        *,
        evaluator: Callable[[ArchitectureVariant], ArchitectureMetrics],
        detector: BottleneckDetector | None = None,
        mutator: ArchitectureMutator | None = None,
        utility: ArchitectureUtility | None = None,
        gate: ArchitecturePromotionGate | None = None,
        latency_budget_ms: float | None = None,
        memory_budget_mb: float | None = None,
    ):
        self.evaluator = evaluator
        self.detector = detector or BottleneckDetector()
        self.mutator = mutator or ArchitectureMutator()
        self.utility = utility or ArchitectureUtility()
        self.gate = gate or ArchitecturePromotionGate()
        self.latency_budget_ms = latency_budget_ms
        self.memory_budget_mb = memory_budget_mb

    def evaluate(self, variant: ArchitectureVariant, *, baseline_metrics: ArchitectureMetrics | None = None) -> ArchitectureEvaluation:
        metrics = self.evaluator(variant)
        utility = self.utility.score(metrics, baseline=baseline_metrics)
        return ArchitectureEvaluation(variant, metrics, utility, fingerprint_variant(variant))

    def optimize(
        self,
        baseline: ArchitectureVariant,
        *,
        rounds: int = 3,
        candidates_per_round: int = 4,
    ) -> OptimizationResult:
        if rounds <= 0 or candidates_per_round <= 0:
            raise ValueError("rounds and candidates_per_round must be positive")
        current = self.evaluate(baseline)
        history = [current]
        rounds_out: list[OptimizationRound] = []

        for _ in range(rounds):
            bottleneck = self.detector.diagnose(
                current.metrics,
                latency_budget_ms=self.latency_budget_ms,
                memory_budget_mb=self.memory_budget_mb,
            )
            candidates = tuple(
                self.evaluate(candidate, baseline_metrics=current.metrics)
                for candidate in self.mutator.mutate(
                    current.variant, bottleneck, count=candidates_per_round
                )
            )
            if not candidates:
                decision = PromotionDecision(
                    False, "no architectural mutations were generated",
                    current.utility, current.utility, None,
                )
                rounds_out.append(OptimizationRound(current, (), bottleneck, decision))
                break

            eligible = [candidate for candidate in candidates if self.gate.accept(current, candidate)]
            if eligible:
                selected = max(eligible, key=lambda item: (item.utility, item.fingerprint))
                decision = PromotionDecision(
                    True,
                    "candidate passed architecture regression gate",
                    current.utility,
                    selected.utility,
                    selected.variant.variant_id,
                )
                current = selected
                history.append(current)
            else:
                decision = PromotionDecision(
                    False,
                    "no candidate passed the architecture regression gate",
                    current.utility,
                    max(candidate.utility for candidate in candidates),
                    None,
                )
            rounds_out.append(OptimizationRound(history[-1] if decision.promoted else current, candidates, bottleneck, decision))

            if not decision.promoted:
                break

        return OptimizationResult(current, tuple(rounds_out), tuple(history))
