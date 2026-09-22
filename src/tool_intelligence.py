"""AI-oriented tool intelligence layer.

Provides capability representations, precondition checks, utility scoring,
contextual tool selection, failure diagnosis and safe fallbacks. The scoring
interfaces are intentionally model-agnostic so a learned policy can replace
the heuristic scorer later without changing the tool boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, sqrt
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True)
class ToolCapability:
    name: str
    description: str
    capabilities: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    risk: float = 0.0
    latency: float = 1.0
    reliability: float = 1.0
    cost: float = 1.0
    fallback_tools: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("tool name must not be empty")
        for value in (self.risk, self.latency, self.cost):
            if value < 0:
                raise ValueError("risk, latency and cost must be non-negative")
        if not 0 <= self.reliability <= 1:
            raise ValueError("reliability must be in [0, 1]")


@dataclass(frozen=True)
class ToolContext:
    goal: str
    required_capabilities: tuple[str, ...] = ()
    available_facts: Mapping[str, Any] = field(default_factory=dict)
    preferred_latency: float | None = None
    risk_tolerance: float = 0.5


@dataclass(frozen=True)
class PreconditionResult:
    satisfied: bool
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolScore:
    tool: str
    score: float
    capability_coverage: float
    reliability: float
    risk_penalty: float
    latency_penalty: float
    cost_penalty: float
    explanation: str


@dataclass(frozen=True)
class ToolSelection:
    selected: str | None
    ranked: tuple[ToolScore, ...]
    reason: str


@dataclass(frozen=True)
class ToolFailureDiagnosis:
    tool: str
    category: str
    retryable: bool
    confidence: float
    fallback: str | None
    explanation: str


@dataclass(frozen=True)
class ToolAttempt:
    tool: str
    success: bool
    error: str | None = None
    latency: float = 0.0


class CapabilityRegistry:
    def __init__(self):
        self._items: dict[str, ToolCapability] = {}

    def register(self, capability: ToolCapability) -> None:
        self._items[capability.name] = capability

    def get(self, name: str) -> ToolCapability | None:
        return self._items.get(name)

    def all(self) -> tuple[ToolCapability, ...]:
        return tuple(self._items.values())


class PreconditionChecker:
    """Checks simple fact predicates; complex predicates can be injected."""
    def __init__(self, predicates: Mapping[str, Callable[[Mapping[str, Any]], bool]] | None = None):
        self.predicates = dict(predicates or {})

    def check(self, tool: ToolCapability, facts: Mapping[str, Any]) -> PreconditionResult:
        missing = []
        for condition in tool.preconditions:
            predicate = self.predicates.get(condition)
            if predicate is not None:
                ok = bool(predicate(facts))
            else:
                ok = bool(facts.get(condition, False))
            if not ok:
                missing.append(condition)
        return PreconditionResult(not missing, tuple(missing))


class ToolUtilityScorer:
    """Contextual utility model with smooth penalties rather than hard ranking rules."""
    def score(self, tool: ToolCapability, context: ToolContext, precondition: PreconditionResult) -> ToolScore:
        required = set(context.required_capabilities)
        offered = set(tool.capabilities)
        coverage = 1.0 if not required else len(required & offered) / len(required)
        reliability = tool.reliability
        risk_penalty = max(0.0, tool.risk - context.risk_tolerance)
        latency_penalty = 0.0
        if context.preferred_latency is not None:
            latency_penalty = max(0.0, tool.latency - context.preferred_latency) / max(context.preferred_latency, 1e-9)
        cost_penalty = tool.cost / (1.0 + tool.cost)
        missing_penalty = 1.0 if not precondition.satisfied else 0.0
        # Logistic utility keeps scores bounded and makes future learned weights easy to substitute.
        raw = 2.5 * coverage + 1.5 * reliability - 2.0 * risk_penalty - latency_penalty - cost_penalty - 4.0 * missing_penalty
        utility = 1.0 / (1.0 + exp(-raw))
        return ToolScore(tool.name, utility, coverage, reliability, risk_penalty, latency_penalty, cost_penalty,
                         f"coverage={coverage:.2f}, reliability={reliability:.2f}, risk={risk_penalty:.2f}, latency={latency_penalty:.2f}, cost={cost_penalty:.2f}")


class ToolSelector:
    def __init__(self, registry: CapabilityRegistry, checker: PreconditionChecker | None = None, scorer: ToolUtilityScorer | None = None):
        self.registry = registry
        self.checker = checker or PreconditionChecker()
        self.scorer = scorer or ToolUtilityScorer()

    def select(self, context: ToolContext) -> ToolSelection:
        scores = []
        for tool in self.registry.all():
            precondition = self.checker.check(tool, context.available_facts)
            scores.append(self.scorer.score(tool, context, precondition))
        ranked = tuple(sorted(scores, key=lambda item: (-item.score, item.tool)))
        usable = tuple(item for item in ranked if item.capability_coverage > 0 and self.checker.check(self.registry.get(item.tool), context.available_facts).satisfied)
        selected = usable[0].tool if usable else None
        return ToolSelection(selected, ranked, "selected highest-utility tool satisfying required preconditions" if selected else "no tool satisfies capability and precondition requirements")


class FailureDiagnoser:
    def diagnose(self, attempt: ToolAttempt, capability: ToolCapability | None = None) -> ToolFailureDiagnosis:
        if attempt.success:
            return ToolFailureDiagnosis(attempt.tool, "none", False, 1.0, None, "tool completed successfully")
        message = (attempt.error or "unknown failure").lower()
        if any(token in message for token in ("timeout", "timed out", "temporarily", "busy")):
            category, retryable, confidence = "transient", True, 0.92
        elif any(token in message for token in ("permission", "denied", "forbidden", "not allowed")):
            category, retryable, confidence = "authorization", False, 0.95
        elif any(token in message for token in ("invalid", "missing", "not found", "precondition")):
            category, retryable, confidence = "precondition", False, 0.90
        else:
            category, retryable, confidence = "execution", True, 0.55
        fallback = None
        if capability:
            fallback = capability.fallback_tools[0] if capability.fallback_tools else None
        return ToolFailureDiagnosis(attempt.tool, category, retryable, confidence, fallback, f"classified failure as {category}")


class SafeToolFallback:
    def __init__(self, selector: ToolSelector, diagnoser: FailureDiagnoser | None = None):
        self.selector = selector
        self.diagnoser = diagnoser or FailureDiagnoser()

    def choose(self, attempt: ToolAttempt, context: ToolContext) -> ToolFailureDiagnosis:
        capability = self.selector.registry.get(attempt.tool)
        diagnosis = self.diagnoser.diagnose(attempt, capability)
        if diagnosis.fallback:
            fallback = self.selector.registry.get(diagnosis.fallback)
            if fallback and self.selector.checker.check(fallback, context.available_facts).satisfied:
                return diagnosis
        selection = self.selector.select(context)
        if selection.selected and selection.selected != attempt.tool:
            return ToolFailureDiagnosis(diagnosis.tool, diagnosis.category, diagnosis.retryable,
                                        diagnosis.confidence, selection.selected,
                                        diagnosis.explanation + f"; safe alternative selected: {selection.selected}")
        return diagnosis


class ToolIntelligence:
    """High-level API used by TARA's brain/tool controller boundary."""
    def __init__(self, registry: CapabilityRegistry | None = None):
        self.registry = registry or CapabilityRegistry()
        self.selector = ToolSelector(self.registry)
        self.fallback = SafeToolFallback(self.selector)

    def register(self, capability: ToolCapability) -> None:
        self.registry.register(capability)

    def choose(self, goal: str, required_capabilities: Iterable[str], *, facts: Mapping[str, Any] | None = None,
               preferred_latency: float | None = None, risk_tolerance: float = 0.5) -> ToolSelection:
        return self.selector.select(ToolContext(goal, tuple(required_capabilities), facts or {}, preferred_latency, risk_tolerance))

    def recover(self, attempt: ToolAttempt, goal: str, required_capabilities: Iterable[str], *, facts: Mapping[str, Any] | None = None,
                risk_tolerance: float = 0.5) -> ToolFailureDiagnosis:
        context = ToolContext(goal, tuple(required_capabilities), facts or {}, risk_tolerance=risk_tolerance)
        return self.fallback.choose(attempt, context)
