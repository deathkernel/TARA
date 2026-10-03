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
from .cognition.global_workspace import GlobalWorkspace, WorkspaceItem
from .cognition.neuromodulation import NeuromodulatoryController, NeuromodulatoryState
from .cognition.executive import ActionCandidate, Decision, ExecutiveController, Goal
from .cognition.metacognition import MetacognitiveMonitor, MetacognitiveReport
from .cognition.reflection import Reflection, ReflectionEngine
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
    reflection: Reflection | None
    neuromodulation: NeuromodulatoryState


class TaraMind:
    """Compose TARA's current computational cognition subsystems."""

    def __init__(self) -> None:
        self.state = CognitiveState()
        self.attention = SelectiveAttention()
        self.workspace = GlobalWorkspace()
        self.neuromodulation = NeuromodulatoryController()
        self.appraisal = AppraisalEngine()
        self.metacognition = MetacognitiveMonitor()
        self.reflection = ReflectionEngine()
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
            self.workspace.broadcast(WorkspaceItem(item.item.content, "attention", item.score))

        appraisal = self.appraisal.evaluate(appraisal_input)
        self.state.affect.valence = appraisal.valence
        self.state.affect.arousal = appraisal.arousal
        self.state.affect.urgency = appraisal.urgency
        self.state.affect.confidence = appraisal.confidence

        world_update = self.world.observe(*transition) if transition else None
        neuro = self.neuromodulation.update(
            novelty=appraisal_input.novelty,
            reward=appraisal_input.goal_congruence,
            surprise=world_update.surprise if world_update else 0.0,
        )
        reflection = (
            self.reflection.compare_states(
                world_update.predicted_state,
                world_update.observation.next_state,
                appraisal.confidence,
            )
            if world_update
            else None
        )
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
        return MindStep(attended, appraisal, report, world_update, decision, plan, reflection, neuro)

    def remember_episode(self, episode: Episode) -> None:
        self.memory.remember(episode)

    def consolidate_memory(self, limit: int = 32):
        return self.memory.consolidate(limit)
