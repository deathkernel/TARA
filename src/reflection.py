"""Basic reflection and experience tracking for TARA Phase 17.

This layer is deliberately deterministic: it records outcomes, identifies
simple success/failure signals, and produces compact reflection notes. It does
not claim human-like self-awareness or autonomous learning.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass(frozen=True)
class Experience:
    goal: str
    action: str
    outcome: str
    success: bool
    score: float = 0.0
    feedback: str = ""


@dataclass(frozen=True)
class Reflection:
    goal: str
    attempts: int
    successes: int
    failures: int
    success_rate: float
    lesson: str
    next_action: str


class ExperienceMemory:
    """Small append-only experience log with optional JSONL persistence."""

    def __init__(self, path=None, max_records=256):
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self.path = Path(path) if path is not None else None
        self.max_records = int(max_records)
        self._records = []
        if self.path is not None and self.path.exists():
            self.load()

    def add(self, experience):
        if not isinstance(experience, Experience):
            raise TypeError("experience must be an Experience")
        self._records.append(experience)
        self._records = self._records[-self.max_records :]
        return experience

    def records(self, goal=None):
        if goal is None:
            return list(self._records)
        return [item for item in self._records if item.goal == goal]

    def save(self, path=None):
        destination = Path(path) if path is not None else self.path
        if destination is None:
            raise ValueError("a path is required to save experiences")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "\n".join(json.dumps(asdict(item), sort_keys=True) for item in self._records)
            + ("\n" if self._records else ""),
            encoding="utf-8",
        )
        self.path = destination

    def load(self, path=None):
        source = Path(path) if path is not None else self.path
        if source is None:
            raise ValueError("a path is required to load experiences")
        records = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(Experience(**payload))
        self._records = records[-self.max_records :]
        self.path = source


class Reflector:
    """Turn recorded outcomes into a compact deterministic reflection."""

    def reflect(self, goal, experiences):
        items = [item for item in experiences if item.goal == goal]
        if not items:
            return Reflection(goal, 0, 0, 0, 0.0, "No experience recorded.", "Attempt the goal and record the outcome.")

        successes = sum(item.success for item in items)
        failures = len(items) - successes
        rate = successes / len(items)
        failed_feedback = [item.feedback.strip() for item in items if not item.success and item.feedback.strip()]
        if successes == len(items):
            lesson = "The recorded approach succeeded consistently."
            next_action = "Reuse the successful approach and verify the next result."
        elif failures == len(items):
            lesson = failed_feedback[-1] if failed_feedback else "The recorded approach has not succeeded yet."
            next_action = "Change the approach, then verify the new result."
        else:
            lesson = failed_feedback[-1] if failed_feedback else "Results are mixed; compare successful and failed attempts."
            next_action = "Prefer the verified successful pattern and test it again."

        return Reflection(goal, len(items), successes, failures, rate, lesson, next_action)
