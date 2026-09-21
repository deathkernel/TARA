"""Small deterministic agent-loop orchestration for TARA.

Research basis:
- ReAct (Yao et al., 2023) interleaves reasoning with observations.
- TARA keeps the loop explicit and dependency-free: observe, remember,
  select the next planned task, accept an externally supplied result, verify,
  and continue or recover.

This module does not execute PC actions, shell commands, browser operations,
or keyboard/mouse input. It is the cognitive control loop that will later be
able to sit above a separately permissioned tool layer.
"""

from dataclasses import dataclass

from .integration import TARAEngine
from .memory import LongTermMemory
from .reasoning import replan, verify_step


@dataclass(frozen=True)
class AgentEvent:
    """One observable event emitted by the cognitive loop."""

    kind: str
    task: str | None = None
    value: object = None


class AgentLoop:
    """Drive TARA's plan from observations and externally supplied results."""

    def __init__(self, engine=None, memory=None):
        self.engine = TARAEngine() if engine is None else engine
        self.memory = LongTermMemory() if memory is None else memory
        self.history = []

    def _emit(self, kind, task=None, value=None):
        event = AgentEvent(kind, task, value)
        self.history.append(event)
        return event

    def observe(self, observation, *, remember_key=None, importance=1.0):
        """Observe an input and optionally persist it in long-term memory."""
        self.engine.observe(observation)
        if remember_key is not None:
            self.memory.remember(remember_key, observation, importance=importance)
            self._emit("remember", value={"key": remember_key, "value": observation})
        return self._emit("observe", value=observation)

    def recall(self, query, limit=None, min_score=0.0):
        """Retrieve relevant remembered context for the current reasoning step."""
        results = self.memory.retrieve(query, limit=limit, min_score=min_score)
        return self._emit("recall", value=results)

    def next_task(self):
        plan = self.engine.state.plan
        if plan is None:
            raise ValueError("a plan must be created before running the agent")
        if plan.complete:
            return self._emit("complete")
        return self._emit("task", task=plan.current.description)

    def submit_result(self, observed, expected):
        """Accept an externally produced result and advance only if verified."""
        plan = self.engine.state.plan
        if plan is None or plan.complete:
            raise ValueError("there is no active task")

        task = plan.current.description
        passed = verify_step(expected, observed)
        self.engine.record_result(observed)
        self._emit("verify", task=task, value=passed)
        if passed:
            plan.advance()
            return self._emit("advance", task=task)
        return self._emit("failure", task=task, value=observed)

    def recover(self, replacement_tasks):
        """Replace the failed task and future path with an explicit alternative."""
        plan = self.engine.state.plan
        if plan is None or plan.complete:
            raise ValueError("there is no failed active task")
        failed_index = plan.current_index
        replan(plan, failed_index, replacement_tasks)
        return self._emit("replan", task=plan.current.description)

    @property
    def complete(self):
        plan = self.engine.state.plan
        return plan is not None and plan.complete
