"""Unified, inspectable cognitive system for TARA Baby.

This is the integration layer around the neural core. It connects attention,
working memory, world modeling, appraisal, executive control, planning and
long-term memory. It intentionally does not claim to be a complete human brain
or consciousness implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cognition.appraisal import AppraisalEngine, AppraisalInput, AppraisalState
from .cognition.attention import AttentionItem, AttendedItem, SelectiveAttention
from .cognition.executive import ActionCandidate, Decision, ExecutiveController, Goal
from .cognition.metacognition import MetacognitiveMonitor, MetacognitiveReport
from .cognition.planning import ModelBasedPlanner, Plan
from .cognition.state import CognitiveState
from .cognition.world_model import WorldModel, WorldUpdate
from .memory.episodic import Episode
from .memory.system import MemorySystem


@dataclass(frozen=True)
class MindStep:
    attended: tuple[AttendedItem, ...]
    appraisal: AppraisalState
    metacognition: MetacognitiveReport
    world_update: WorldUpdate | None
    decision: Decision
    plan: Plan | None


class TaraMind:
    """Compose TARA's current computational cognition subsystems."""

    def __init__(self) -> None:
        self.state = CognitiveState()
        self.attention = SelectiveAttention()
        self.appraisal = AppraisalEngine()
        self.metacognition = MetacognitiveMonitor()
        self.executive = ExecutiveController()
        self.world = WorldModel()
        self.memory = MemorySystem()

    def set_goal(self, goal_id: str, description: str, priority: float = 0.8) -> None:
        self.executive.add_goal(Goal(goal_id, description, priority))
        self.state.goal = description

    def process(
        self,
        text: str,
        attention_items: list[AttentionItem],
        appraisal_input: AppraisalInput,
        actions: list[ActionCandidate] | None = None,
        transition: tuple[str, str] | None = None,
        plan_goal: str | None = None,
    ) -> MindStep:
        if not text.strip():
            raise ValueError("text must be non-empty")
        attended = tuple(self.attention.allocate(attention_items, budget=7))
        self.state.observe(text)
        for item in attended:
            self.state.remember_working(item.item.content, source="attention", importance=item.score)

        appraisal = self.appraisal.evaluate(appraisal_input)
        self.state.affect.valence = appraisal.valence
        self.state.affect.arousal = appraisal.arousal
        self.state.affect.urgency = appraisal.urgency
        self.state.affect.confidence = appraisal.confidence

        world_update = self.world.observe(*transition) if transition else None
        decision = self.executive.choose(actions or [])
        report = self.metacognition.assess(
            base_confidence=appraisal.confidence,
            evidence_count=len(self.state.observations),
            prediction_error=world_update.surprise if world_update else 0.0,
            ambiguity=1.0 - appraisal.confidence,
            consequence_cost=min(1.0, max(0.0, decision.score)) if decision.blocked else 0.0,
        )
        plan = None
        if plan_goal and self.world.current_state:
            plan = ModelBasedPlanner(self.world.predictive_map).plan(
                self.world.current_state, plan_goal
            )
        return MindStep(attended, appraisal, report, world_update, decision, plan)

    def remember_episode(self, episode: Episode) -> None:
        self.memory.remember(episode)

    def consolidate_memory(self, limit: int = 32):
        return self.memory.consolidate(limit)
