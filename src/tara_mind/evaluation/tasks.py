"""Task schema for capability evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class EvaluationTask:
    task_id: str
    category: str
    prompt: str
    scorer: Callable[[str], float]

    def score(self, output: str) -> float:
        value = float(self.scorer(output))
        if not 0.0 <= value <= 1.0:
            raise ValueError("evaluation score must be in [0, 1]")
        return value
