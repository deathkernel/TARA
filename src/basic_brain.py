"""Complete basic cognitive architecture for TARA.

This module provides one coherent, dependency-light baseline for the full
cognitive loop. It intentionally favors explicit state and deterministic
behavior over sophisticated models; each layer can later be upgraded without
changing the public brain boundary.

Layers:
    perception -> working memory -> long-term memory -> reasoning -> planning
    -> tools/actions -> observation -> verification -> reflection -> learning
    -> hypotheses -> experiments -> knowledge graph -> research -> multimodal
    -> autonomous orchestration

The baseline does not perform unrestricted PC automation or execute arbitrary
model-generated code. Tools are explicitly registered by the host application.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Callable, Iterable, Mapping


# ---------------------------------------------------------------------------
# Perception


@dataclass(frozen=True)
class Percept:
    kind: str
    content: Any
    source: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Perception:
    """Normalize raw inputs into a common percept representation."""

    def text(self, value: str, source: str = "text") -> Percept:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("text perception must be a non-empty string")
        return Percept("text", value, source)

    def file(self, path: str, content: str | None = None) -> Percept:
        return Percept("file", {"path": path, "content": content}, "file")

    def screen(self, description: str) -> Percept:
        return Percept("screen", description, "screen")

    def audio(self, transcript: str) -> Percept:
        return Percept("audio", transcript, "audio")

    def image(self, description: str) -> Percept:
        return Percept("image", description, "vision")

    def normalize(self, value: Any, source: str = "input") -> Percept:
        if isinstance(value, Percept):
            return value
        if isinstance(value, str):
            return self.text(value, source)
        return Percept("structured", value, source)


# ---------------------------------------------------------------------------
# Memory


@dataclass(frozen=True)
class MemoryItem:
    key: str
    value: Any
    importance: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class WorkingMemory:
    """Bounded short-term memory for the active cognitive cycle."""

    def __init__(self, capacity: int = 16) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: deque[MemoryItem] = deque(maxlen=capacity)

    def add(self, value: Any, *, key: str | None = None, importance: float = 1.0) -> MemoryItem:
        item = MemoryItem(key or f"wm-{len(self._items)}", value, float(importance))
        self._items.append(item)
        return item

    def recent(self, limit: int | None = None) -> tuple[MemoryItem, ...]:
        items = tuple(self._items)
        return items if limit is None else items[-max(0, limit):]

    def clear(self) -> None:
        self._items.clear()


class LongTermMemory:
    """Small persistent-style memory abstraction with deterministic retrieval."""

    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}

    def remember(self, key: str, value: Any, importance: float = 1.0) -> MemoryItem:
        if not key.strip():
            raise ValueError("memory key must not be empty")
        item = MemoryItem(key, value, float(importance))
        self._items[key] = item
        return item

    def retrieve(self, query: str, limit: int = 5) -> tuple[MemoryItem, ...]:
        if limit < 0:
            raise ValueError("limit must not be negative")
        terms = set(query.lower().split())
        scored: list[tuple[float, MemoryItem]] = []
        for item in self._items.values():
            text = f"{item.key} {item.value}".lower()
            overlap = len(terms.intersection(text.split()))
            scored.append((overlap + 0.001 * item.importance, item))
        scored.sort(key=lambda pair: (-pair[0], pair[1].key))
        return tuple(item for score, item in scored[:limit] if score > 0)

    def forget(self, key: str) -> bool:
        return self._items.pop(key, None) is not None

    def all(self) -> tuple[MemoryItem, ...]:
        return tuple(self._items.values())


# ---------------------------------------------------------------------------
# Reasoning and planning


@dataclass(frozen=True)
class Goal:
    description: str
    success_conditions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReasoningResult:
    conclusion: str
    evidence: tuple[Any, ...]
    confidence: float


class Reasoner:
    """Baseline symbolic reasoner; the neural reasoner can replace it later."""

    def infer(self, question: str, evidence: Iterable[Any]) -> ReasoningResult:
        evidence_tuple = tuple(evidence)
        if not question.strip():
            raise ValueError("question must not be empty")
        if not evidence_tuple:
            return ReasoningResult("insufficient evidence", (), 0.0)
        return ReasoningResult(
            f"Based on {len(evidence_tuple)} observation(s), investigate: {question.strip()}",
            evidence_tuple,
            min(0.5 + 0.1 * len(evidence_tuple), 0.9),
        )


@dataclass(frozen=True)
class PlanStep:
    id: str
    description: str


@dataclass
class Plan:
    goal: Goal
    steps: list[PlanStep]
    current: int = 0

    @property
    def done(self) -> bool:
        return self.current >= len(self.steps)

    def next_step(self) -> PlanStep | None:
        return None if self.done else self.steps[self.current]

    def advance(self) -> PlanStep | None:
        if not self.done:
            self.current += 1
        return self.next_step()


class Planner:
    def make_plan(self, goal: Goal, subtasks: Iterable[str]) -> Plan:
        steps = [PlanStep(f"step-{i}", task.strip()) for i, task in enumerate(subtasks, 1) if task.strip()]
        if not steps:
            raise ValueError("a plan requires at least one non-empty step")
        return Plan(goal, steps)


# ---------------------------------------------------------------------------
# Tools / actions


@dataclass(frozen=True)
class ToolResult:
    tool: str
    success: bool
    output: Any = None
    error: str | None = None


class ToolRegistry:
    """Explicit allow-list of host-provided actions."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, function: Callable[..., Any]) -> None:
        if not name.strip():
            raise ValueError("tool name must not be empty")
        self._tools[name] = function

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def execute(self, name: str, **kwargs: Any) -> ToolResult:
        function = self._tools.get(name)
        if function is None:
            return ToolResult(name, False, error="tool is not registered")
        try:
            return ToolResult(name, True, output=function(**kwargs))
        except Exception as exc:  # host tools are untrusted boundaries
            return ToolResult(name, False, error=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Verification / reflection / learning


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    expected: Any
    observed: Any
    reason: str


class Verifier:
    def check(self, observed: Any, expected: Any) -> VerificationResult:
        passed = observed == expected
        return VerificationResult(
            passed,
            expected,
            observed,
            "match" if passed else "observed result differs from expectation",
        )


@dataclass(frozen=True)
class Reflection:
    summary: str
    lessons: tuple[str, ...]
    success: bool


class Reflector:
    def reflect(self, verification: VerificationResult) -> Reflection:
        if verification.passed:
            return Reflection("action succeeded", ("retain the verified approach",), True)
        return Reflection(
            "action failed verification",
            ("inspect the failure", "change the plan or action", "verify again"),
            False,
        )


@dataclass(frozen=True)
class LearningRecord:
    fingerprint: str
    input: Any
    output: Any
    verified: bool
    lesson: str


class LearningStore:
    """Keeps verified outcomes separate from speculative outcomes."""

    def __init__(self) -> None:
        self._records: dict[str, LearningRecord] = {}

    def learn(self, input_value: Any, output: Any, verification: VerificationResult, lesson: str) -> LearningRecord | None:
        if not verification.passed:
            return None
        fingerprint = sha256(repr((input_value, output)).encode("utf-8")).hexdigest()
        record = LearningRecord(fingerprint, input_value, output, True, lesson)
        self._records.setdefault(fingerprint, record)
        return record

    def all(self) -> tuple[LearningRecord, ...]:
        return tuple(self._records.values())


# ---------------------------------------------------------------------------
# Hypothesis / experiment / knowledge graph


@dataclass(frozen=True)
class Hypothesis:
    statement: str
    evidence: tuple[Any, ...] = ()


@dataclass(frozen=True)
class ExperimentResult:
    hypothesis: Hypothesis
    observed: Any
    supported: bool


class HypothesisEngine:
    def propose(self, question: str, evidence: Iterable[Any]) -> tuple[Hypothesis, ...]:
        evidence_tuple = tuple(evidence)
        return (
            Hypothesis(f"A possible explanation for '{question.strip()}' is related to the available evidence.", evidence_tuple),
            Hypothesis(f"An alternative explanation for '{question.strip()}' should be tested.", evidence_tuple),
        )


class ExperimentEngine:
    def run(self, hypothesis: Hypothesis, test: Callable[[Hypothesis], Any]) -> ExperimentResult:
        observed = test(hypothesis)
        return ExperimentResult(hypothesis, observed, bool(observed))


@dataclass(frozen=True)
class KnowledgeEdge:
    subject: str
    relation: str
    object: str


class KnowledgeGraph:
    def __init__(self) -> None:
        self._edges: set[KnowledgeEdge] = set()

    def add(self, subject: str, relation: str, object: str) -> KnowledgeEdge:
        edge = KnowledgeEdge(subject, relation, object)
        self._edges.add(edge)
        return edge

    def related(self, subject: str) -> tuple[KnowledgeEdge, ...]:
        return tuple(sorted((edge for edge in self._edges if edge.subject == subject), key=lambda edge: (edge.relation, edge.object)))

    def all(self) -> tuple[KnowledgeEdge, ...]:
        return tuple(sorted(self._edges, key=lambda edge: (edge.subject, edge.relation, edge.object)))


# ---------------------------------------------------------------------------
# Research / multimodal boundary


class ResearchInterface:
    """Host-injected research provider; no implicit network access."""

    def __init__(self, provider: Callable[[str], Iterable[Any]] | None = None) -> None:
        self.provider = provider

    def search(self, query: str) -> tuple[Any, ...]:
        if self.provider is None:
            return ()
        return tuple(self.provider(query))


@dataclass(frozen=True)
class MultimodalInput:
    text: tuple[str, ...] = ()
    images: tuple[Any, ...] = ()
    audio: tuple[Any, ...] = ()
    files: tuple[Any, ...] = ()


class MultimodalLayer:
    def pack(self, *, text: Iterable[str] = (), images: Iterable[Any] = (), audio: Iterable[Any] = (), files: Iterable[Any] = ()) -> MultimodalInput:
        return MultimodalInput(tuple(text), tuple(images), tuple(audio), tuple(files))


# ---------------------------------------------------------------------------
# Complete basic brain


@dataclass(frozen=True)
class CycleResult:
    percept: Percept
    reasoning: ReasoningResult
    plan: Plan
    tool_result: ToolResult | None
    verification: VerificationResult | None
    reflection: Reflection | None
    learned: bool


class BasicTARABrain:
    """One connected baseline implementation of TARA's complete cognitive loop."""

    def __init__(self, *, working_memory_capacity: int = 16, tool_registry: ToolRegistry | None = None) -> None:
        self.perception = Perception()
        self.working_memory = WorkingMemory(working_memory_capacity)
        self.long_term_memory = LongTermMemory()
        self.reasoner = Reasoner()
        self.planner = Planner()
        self.tools = tool_registry or ToolRegistry()
        self.verifier = Verifier()
        self.reflector = Reflector()
        self.learning = LearningStore()
        self.hypotheses = HypothesisEngine()
        self.experiments = ExperimentEngine()
        self.knowledge = KnowledgeGraph()
        self.research = ResearchInterface()
        self.multimodal = MultimodalLayer()
        self.goal: Goal | None = None
        self.plan: Plan | None = None

    def observe(self, value: Any, *, source: str = "input", remember: bool = True) -> Percept:
        percept = self.perception.normalize(value, source)
        self.working_memory.add(percept)
        if remember:
            key = f"observation-{len(self.long_term_memory.all()) + 1}"
            self.long_term_memory.remember(key, percept)
        return percept

    def set_goal(self, description: str, success_conditions: Iterable[str] = ()) -> Goal:
        self.goal = Goal(description.strip(), tuple(success_conditions))
        self.plan = None
        return self.goal

    def make_plan(self, subtasks: Iterable[str]) -> Plan:
        if self.goal is None:
            raise ValueError("set a goal before making a plan")
        self.plan = self.planner.make_plan(self.goal, subtasks)
        return self.plan

    def reason(self, question: str) -> ReasoningResult:
        evidence = self.working_memory.recent()
        return self.reasoner.infer(question, evidence)

    def act(self, tool: str, **kwargs: Any) -> ToolResult:
        return self.tools.execute(tool, **kwargs)

    def verify(self, observed: Any, expected: Any) -> VerificationResult:
        return self.verifier.check(observed, expected)

    def reflect(self, verification: VerificationResult) -> Reflection:
        return self.reflector.reflect(verification)

    def run_cycle(
        self,
        value: Any,
        *,
        goal: str,
        subtasks: Iterable[str],
        tool: str | None = None,
        tool_kwargs: Mapping[str, Any] | None = None,
        expected: Any = None,
        verify: bool = False,
    ) -> CycleResult:
        percept = self.observe(value)
        self.set_goal(goal)
        plan = self.make_plan(subtasks)
        reasoning = self.reason(goal)
        tool_result = self.act(tool, **dict(tool_kwargs or {})) if tool else None
        verification = self.verify(tool_result.output if tool_result and tool_result.success else tool_result, expected) if verify else None
        reflection = self.reflect(verification) if verification else None
        learned_record = self.learning.learn(value, tool_result.output if tool_result else reasoning.conclusion, verification, reflection.summary if reflection else "") if verification else None
        if verification and verification.passed:
            plan.advance()
        return CycleResult(percept, reasoning, plan, tool_result, verification, reflection, learned_record is not None)

    def snapshot(self) -> dict[str, Any]:
        return {
            "goal": None if self.goal is None else self.goal.description,
            "plan": None if self.plan is None else {
                "steps": [step.description for step in self.plan.steps],
                "current": self.plan.current,
                "done": self.plan.done,
            },
            "working_memory": [item.value for item in self.working_memory.recent()],
            "long_term_memory": len(self.long_term_memory.all()),
            "learned_records": len(self.learning.all()),
            "knowledge_edges": len(self.knowledge.all()),
            "registered_tools": self.tools.names(),
        }


__all__ = [
    "BasicTARABrain", "CycleResult", "Percept", "Perception", "MemoryItem",
    "WorkingMemory", "LongTermMemory", "Goal", "ReasoningResult", "Reasoner",
    "PlanStep", "Plan", "Planner", "ToolResult", "ToolRegistry",
    "VerificationResult", "Verifier", "Reflection", "Reflector", "LearningRecord",
    "LearningStore", "Hypothesis", "HypothesisEngine", "ExperimentResult",
    "ExperimentEngine", "KnowledgeEdge", "KnowledgeGraph", "ResearchInterface",
    "MultimodalInput", "MultimodalLayer",
]
