"""Inspectible cognitive-loop orchestration for TARA Baby."""

from __future__ import annotations

from dataclasses import dataclass

from .state import CognitiveState


@dataclass(frozen=True)
class CognitiveStep:
    """A trace entry describing one transition in the cognitive loop."""

    stage: str
    detail: str


class CognitiveLoop:
    """A deterministic orchestration shell around the future learned planner.

    The loop itself does not pretend to provide intelligence. It defines the
    state transitions that learned components and tools will eventually fill.
    """

    STAGES = (
        "perception",
        "attention",
        "working_memory",
        "world_state",
        "reasoning",
        "appraisal",
        "executive_control",
        "planning",
        "action",
        "observation",
        "reflection",
        "memory",
    )

    def __init__(self, state: CognitiveState | None = None):
        self.state = state or CognitiveState()
        self.trace: list[CognitiveStep] = []

    def record(self, stage: str, detail: str) -> None:
        if stage not in self.STAGES:
            raise ValueError(f"unknown cognitive stage: {stage}")
        if not isinstance(detail, str) or not detail.strip():
            raise ValueError("detail must be a non-empty string")
        self.trace.append(CognitiveStep(stage=stage, detail=detail.strip()))

    def reset_trace(self) -> None:
        self.trace.clear()
