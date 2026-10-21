"""Unified cognitive loop for TARA.

The loop coordinates existing subsystems but keeps external execution explicit.
Each cycle records perception, recall, structured reasoning, plan selection,
tool choice, action outcome, verification, reflection, learning and progress.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Callable, Iterable, Mapping

from .advanced_planning import AdaptivePlan, AdvancedPlanner, PlanStep, StepStatus
from .experience_learning import Experience, ExperienceLearningEngine, LearningReport
from .reflection_loop import ReflectionLoop
from .goal_progress import GoalProgress, Progress
from .structured_reasoning import ReasoningState, StructuredReasoner
from .tool_intelligence import ToolAttempt, ToolSelection, ToolIntelligence


@dataclass(frozen=True)
class ActionRequest:
    cycle_id: str
    step: PlanStep
    tool: ToolSelection | None
    perception: str
    recalled: tuple[Any, ...]


@dataclass(frozen=True)
class ActionOutcome:
    observed: Any
    success: bool
    score: float = 0.0
    cost: float = 0.0
    feedback: str = ""
    metadata: Mapping[str, str] = None


@dataclass(frozen=True)
class CycleTrace:
    cycle_id: str
    goal: str
    perception: str
    recalled: tuple[Any, ...]
    reasoning: ReasoningState
    step_id: str
    selected_tool: str | None
    outcome: ActionOutcome
    verified: bool
    reflection: object
    learning: LearningReport
    progress: Progress
    next_step_id: str | None
    status: str


@dataclass(frozen=True)
class CognitiveLoopReport:
    goal: str
    traces: tuple[CycleTrace, ...]
    completed: bool
    stopped: bool
    fingerprint: str


def _fingerprint(parts: Iterable[str]) -> str:
    return sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


class UnifiedCognitiveLoop:
    def __init__(
        self,
        *,
        reasoner: StructuredReasoner | None = None,
        planner: AdvancedPlanner | None = None,
        tools: ToolIntelligence | None = None,
        memory=None,
        reflection: ReflectionLoop | None = None,
        progress: GoalProgress | None = None,
        learning: ExperienceLearningEngine | None = None,
        max_cycles: int = 32,
    ) -> None:
        if max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        if memory is None:
            raise ValueError("memory is required")
        self.reasoner = reasoner or StructuredReasoner()
        self.planner = planner or AdvancedPlanner()
        self.tools = tools or ToolIntelligence()
        self.memory = memory
        self.reflection = reflection or ReflectionLoop()
        self.progress = progress or GoalProgress()
        self.learning = learning or ExperienceLearningEngine()
        self.max_cycles = max_cycles
        self.plan: AdaptivePlan | None = None
        self.goal: str | None = None
        self.traces: list[CycleTrace] = []
        self.stopped = False

    def start(self, goal: str, subtasks, *, total: int | None = None) -> AdaptivePlan:
        plan = self.planner.create(goal, subtasks)
        self.goal = goal
        self.plan = plan
        self.traces.clear()
        self.stopped = False
        self.progress.start(goal, total or max(1, len(plan.steps)))
        return plan

    def stop(self) -> None:
        self.stopped = True

    def resume(self) -> None:
        self.stopped = False

    def cycle(
        self,
        perception: Any,
        *,
        action_executor: Callable[[ActionRequest], ActionOutcome | Mapping[str, Any] | Any],
        expected: Any = None,
        required_capabilities: Iterable[str] = (),
        tool_facts: Mapping[str, str] | None = None,
        action_tool: str | None = None,
    ) -> CycleTrace:
        if self.plan is None or self.goal is None:
            raise ValueError("start() must be called before cycle()")
        if self.stopped:
            raise RuntimeError("cognitive loop is stopped")
        if len(self.traces) >= self.max_cycles:
            raise RuntimeError("maximum cognitive cycles reached")

        ready = self.plan.ready_steps()
        if not ready:
            raise RuntimeError("no ready plan step remains")
        step = ready[0]
        cycle_id = f"cycle-{len(self.traces):04d}"

        perception_text = str(perception)
        try:
            remembered = tuple(self.memory.retrieve(perception_text, limit=8))
        except Exception:
            remembered = ()

        reasoning = self.reasoner.start(self.goal)
        facts = (perception_text,) + tuple(str(item) for item in remembered)
        self.reasoner.reason(reasoning, facts=facts, hypotheses=(step.description,))

        selection: ToolSelection | None = None
        capabilities = tuple(required_capabilities)
        if capabilities:
            selection = self.tools.choose(
                self.goal,
                capabilities,
                facts=tool_facts,
            )

        request = ActionRequest(cycle_id, step, selection, perception_text, remembered)
        try:
            raw = action_executor(request)
            outcome = self._normalize_outcome(raw)
        except Exception as exc:
            outcome = ActionOutcome(
                observed=None,
                success=False,
                feedback=f"{type(exc).__name__}: {exc}",
            )

        verified = self._verify(outcome, expected)
        step.attempts += 1
        step.result = outcome.observed
        if verified:
            step.status = StepStatus.COMPLETED
        else:
            step.status = StepStatus.FAILED
            step.failure = outcome.feedback or "verification failed"

        key = f"{self.goal}:{cycle_id}"
        try:
            memory_value = {
                "observed": outcome.observed,
                "verified": verified,
                "success": outcome.success,
                "feedback": outcome.feedback,
                "score": outcome.score,
            }
            self.memory.remember(
                key,
                memory_value,
                importance=max(0.1, abs(outcome.score) + 0.5),
            )
        except Exception:
            pass

        reflection_cycle = self.reflection.record(
            self.goal,
            step.description if action_tool is None else action_tool,
            outcome.observed,
            verified,
            outcome.score,
            outcome.feedback,
        )
        experience = Experience(
            cycle_id,
            perception_text,
            step.description if action_tool is None else action_tool,
            str(outcome.observed),
            outcome.score,
            verified,
            outcome.cost,
        )
        learning = self.learning.learn([experience])
        progress = self.progress.mark_complete(self.goal, 1) if verified else self.progress.snapshot(self.goal)
        next_ready = self.plan.ready_steps() if verified else ()
        status = "completed" if self.plan.completed() and len(self.plan.completed()) == len(self.plan.steps) else ("verified" if verified else "failed")
        trace = CycleTrace(
            cycle_id, self.goal, perception_text, remembered, reasoning, step.step_id,
            selection.selected if selection and selection.selected else None,
            outcome, verified, reflection_cycle, learning, progress,
            next_ready[0].step_id if next_ready else None, status,
        )
        self.traces.append(trace)
        return trace

    def run(
        self,
        goal: str,
        subtasks,
        perceptions: Iterable[Any],
        *,
        action_executor: Callable[[ActionRequest], ActionOutcome | Mapping[str, Any] | Any],
        expected: Any = None,
        required_capabilities: Iterable[str] = (),
        tool_facts: Mapping[str, str] | None = None,
    ) -> CognitiveLoopReport:
        self.start(goal, subtasks)
        for perception in perceptions:
            if self.stopped or self.plan is None:
                break
            if not self.plan.remaining():
                break
            try:
                self.cycle(
                    perception,
                    action_executor=action_executor,
                    expected=expected,
                    required_capabilities=required_capabilities,
                    tool_facts=tool_facts,
                )
            except RuntimeError:
                break
        completed = self.plan is not None and not self.plan.remaining()
        fingerprint = _fingerprint([trace.cycle_id + ":" + trace.status for trace in self.traces])
        return CognitiveLoopReport(goal, tuple(self.traces), completed, self.stopped, fingerprint)

    @staticmethod
    def _verify(outcome: ActionOutcome, expected: Any) -> bool:
        if not outcome.success:
            return False
        if expected is None:
            return True
        if callable(expected):
            return bool(expected(outcome.observed))
        return outcome.observed == expected

    @staticmethod
    def _normalize_outcome(raw: ActionOutcome | Mapping[str, Any] | Any) -> ActionOutcome:
        if isinstance(raw, ActionOutcome):
            return raw
        if isinstance(raw, Mapping):
            return ActionOutcome(
                observed=raw.get("observed", raw.get("result")),
                success=bool(raw.get("success", True)),
                score=float(raw.get("score", 0.0)),
                cost=float(raw.get("cost", 0.0)),
                feedback=str(raw.get("feedback", "")),
                metadata=raw.get("metadata") or {},
            )
        return ActionOutcome(raw, True)


__all__ = [
    "ActionRequest",
    "ActionOutcome",
    "CycleTrace",
    "CognitiveLoopReport",
    "UnifiedCognitiveLoop",
]
