"""Top-level TARA Core runtime.

Phase 35 consolidates the existing cognitive subsystems behind one explicit,
bounded runtime. It owns lifecycle, component discovery, health diagnostics,
event tracing and safe shutdown; domain logic remains in the underlying
specialized modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .architecture_optimization import ArchitectureOptimizer, ArchitectureVariant, OptimizationResult
from .autonomous_research import ResearchReport
from .brain import TARABrain
from .experience_learning import Experience, LearningReport
from .scientific_experiment import ExperimentDesign, ExperimentReport
from .unified_cognitive_loop import ActionOutcome, CognitiveLoopReport


class CoreStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True)
class CoreConfig:
    seed: int = 0
    max_cycles: int = 32
    max_events: int = 256

    def __post_init__(self) -> None:
        if self.max_cycles <= 0 or self.max_events <= 0:
            raise ValueError("core bounds must be positive")


@dataclass(frozen=True)
class CoreEvent:
    sequence: int
    kind: str
    payload: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CoreHealth:
    status: CoreStatus
    component_count: int
    missing_components: tuple[str, ...]
    event_count: int
    healthy: bool


@dataclass(frozen=True)
class CoreSnapshot:
    status: str
    goal: str | None
    event_count: int
    cognitive_cycles: int
    components: tuple[str, ...]
    fingerprint: str


def _fingerprint(parts: Iterable[str]) -> str:
    return sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


class TARACore:
    """Single top-level runtime for TARA's cognitive architecture."""

    REQUIRED_COMPONENTS = (
        "brain", "memory", "world", "reasoner", "planner", "tools",
        "researcher", "experimenter", "cognitive_loop", "continual_learning", "multimodal",
    )

    def __init__(self, brain: TARABrain, *, config: CoreConfig | None = None, architecture_optimizer: ArchitectureOptimizer | None = None) -> None:
        if not isinstance(brain, TARABrain):
            raise TypeError("brain must be a TARABrain")
        self.config = config or CoreConfig()
        self.brain = brain
        self.status = CoreStatus.READY
        self._events: list[CoreEvent] = []
        self._goal: str | None = None
        self._sequence = 0
        if architecture_optimizer is not None:
            self.brain.architecture_optimizer = architecture_optimizer
        self.brain.cognitive_loop.max_cycles = self.config.max_cycles

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, config: CoreConfig | None = None, engine=None, memory=None) -> "TARACore":
        brain = TARABrain.from_checkpoint(path, engine=engine, memory=memory, seed=(config or CoreConfig()).seed)
        return cls(brain, config=config)

    @property
    def components(self) -> dict[str, Any]:
        return {
            "brain": self.brain, "memory": self.brain.memory, "world": self.brain.world,
            "reasoner": self.brain.reasoner, "planner": self.brain.planner, "tools": self.brain.tools,
            "researcher": self.brain.researcher, "experimenter": self.brain.experimenter,
            "architecture_optimizer": self.brain.architecture_optimizer,
            "cognitive_loop": self.brain.cognitive_loop, "continual_learning": self.brain.continual,
            "multimodal": self.brain.multimodal,
        }

    def _emit(self, kind: str, **payload: Any) -> CoreEvent:
        self._sequence += 1
        normalized = tuple((str(key), str(value)) for key, value in sorted(payload.items()))
        event = CoreEvent(self._sequence, kind, normalized)
        self._events.append(event)
        if len(self._events) > self.config.max_events:
            del self._events[:-self.config.max_events]
        return event

    def health(self) -> CoreHealth:
        components = self.components
        missing = tuple(name for name in self.REQUIRED_COMPONENTS if components.get(name) is None)
        return CoreHealth(self.status, len(components), missing, len(self._events), self.status != CoreStatus.ERROR and not missing)

    def events(self, *, limit: int | None = None) -> tuple[CoreEvent, ...]:
        if limit is None:
            return tuple(self._events)
        if limit <= 0:
            raise ValueError("limit must be positive")
        return tuple(self._events[-limit:])

    def set_goal(self, goal: str, subtasks, *, total: int | None = None) -> None:
        if self.status == CoreStatus.STOPPED:
            raise RuntimeError("core is stopped")
        self._goal = str(goal)
        self.brain.start_cognitive_loop(self._goal, subtasks, total=total)
        self._emit("goal_started", goal=self._goal)

    def cycle(self, perception: Any, *, action_executor: Callable[[Any], ActionOutcome | Mapping[str, Any] | Any], expected: Any = None, required_capabilities: Iterable[str] = (), tool_facts: Mapping[str, str] | None = None):
        if self.status == CoreStatus.STOPPED:
            raise RuntimeError("core is stopped")
        self.status = CoreStatus.RUNNING
        try:
            trace = self.brain.cognitive_cycle(perception, action_executor=action_executor, expected=expected, required_capabilities=required_capabilities, tool_facts=tool_facts)
            self._emit("cycle", cycle_id=trace.cycle_id, status=trace.status, verified=trace.verified)
            if trace.status == "completed": self.status = CoreStatus.READY
            return trace
        except Exception as exc:
            self.status = CoreStatus.ERROR
            self._emit("error", type=type(exc).__name__, message=str(exc))
            raise

    def run(self, goal: str, subtasks, perceptions: Iterable[Any], *, action_executor: Callable[[Any], ActionOutcome | Mapping[str, Any] | Any], expected: Any = None, required_capabilities: Iterable[str] = (), tool_facts: Mapping[str, str] | None = None) -> CognitiveLoopReport:
        if self.status == CoreStatus.STOPPED:
            raise RuntimeError("core is stopped")
        self.status = CoreStatus.RUNNING
        self._goal = str(goal)
        self._emit("goal_started", goal=self._goal)
        try:
            report = self.brain.run_cognitive_loop(self._goal, subtasks, perceptions, action_executor=action_executor, expected=expected, required_capabilities=required_capabilities, tool_facts=tool_facts)
            self._emit("run_finished", completed=report.completed, cycles=len(report.traces))
            self.status = CoreStatus.READY
            return report
        except Exception as exc:
            self.status = CoreStatus.ERROR
            self._emit("error", type=type(exc).__name__, message=str(exc))
            raise

    def respond(self, prompt: str, **kwargs):
        response = self.brain.respond(prompt, **kwargs)
        self._emit("response", characters=len(response.text))
        return response

    def generate(self, prompt: str, **kwargs) -> str:
        text = self.brain.generate(prompt, **kwargs)
        self._emit("generation", characters=len(text))
        return text

    def evaluate_checkpoint(self, path: str | Path, *, output: str | Path | None = None, max_new_tokens: int = 32):
        from .capability_suite import capability_cases
        from .model_capability_runner import evaluate_checkpoint, write_evaluation
        evaluation = evaluate_checkpoint(path, tuple(capability_cases()), max_new_tokens=max_new_tokens)
        if output is not None:
            write_evaluation(evaluation, output)
        self._emit("capability_evaluation", checkpoint=str(path), score=evaluation.benchmark.overall_score)
        return evaluation

    def perceive(self, observations):
        context = self.brain.perceive_multimodal(observations)
        self._emit("perception", modalities=",".join(item.modality for item in context.observations))
        return context

    def research(self, question: str, searcher, *, max_queries: int | None = None) -> ResearchReport:
        report = self.brain.conduct_research(question, searcher, max_queries=max_queries)
        self._emit("research", confidence=report.overall_confidence, sources=len(report.sources))
        return report

    def experiment(self, design: ExperimentDesign, executor) -> ExperimentReport:
        report = self.brain.run_experiment(design, executor)
        self._emit("experiment", trials=len(report.trials), conclusions=len(report.conclusions))
        return report

    def optimize_architecture(self, baseline: ArchitectureVariant, *, optimizer: ArchitectureOptimizer | None = None, rounds: int = 3, candidates_per_round: int = 4) -> OptimizationResult:
        result = self.brain.optimize_architecture(baseline, rounds=rounds, candidates_per_round=candidates_per_round, optimizer=optimizer)
        self._emit("architecture_optimization", promoted=len(result.history) > 1)
        return result

    def learn(self, experiences: Sequence[Experience]) -> LearningReport:
        report = self.brain.experience_learning.learn(experiences)
        self._emit("learning", experiences=len(experiences), updates=len(report.updates))
        return report

    def build_replay(self, records):
        batch = self.brain.build_replay(records)
        self._emit("replay", examples=len(batch.examples))
        return batch

    def snapshot(self) -> CoreSnapshot:
        cycles = len(self.brain.cognitive_loop.traces)
        components = tuple(sorted(name for name, item in self.components.items() if item is not None))
        fingerprint = _fingerprint([self.status.value, self._goal or "", str(len(self._events)), str(cycles), *components])
        return CoreSnapshot(self.status.value, self._goal, len(self._events), cycles, components, fingerprint)

    def save_runtime_manifest(self, path: str | Path) -> CoreSnapshot:
        snapshot = self.snapshot()
        payload = {"status": snapshot.status, "goal": snapshot.goal, "event_count": snapshot.event_count, "cognitive_cycles": snapshot.cognitive_cycles, "components": snapshot.components, "fingerprint": snapshot.fingerprint, "config": {"seed": self.config.seed, "max_cycles": self.config.max_cycles, "max_events": self.config.max_events}}
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return snapshot

    def stop(self) -> None:
        self.brain.stop_cognitive_loop(); self.brain.stop_tasks(); self.status = CoreStatus.STOPPED; self._emit("stopped")

    def resume(self) -> None:
        self.brain.resume_cognitive_loop(); self.brain.resume_tasks(); self.status = CoreStatus.READY; self._emit("resumed")


__all__ = ["CoreConfig", "CoreEvent", "CoreHealth", "CoreSnapshot", "CoreStatus", "TARACore"]
