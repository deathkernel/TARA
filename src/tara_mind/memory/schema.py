"""Schema learning from repeated episodic structure.

Scientific grounding: schemas are generalized knowledge structures about typical
sequences in a context. Recent cognitive-neuroscience work connects schema
learning to prediction errors, hierarchical learning and simplified abstractions.
TARA implements a small statistical analogue rather than a biological simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict


@dataclass(frozen=True)
class Schema:
    name: str
    context: str
    steps: tuple[str, ...]
    support: int
    confidence: float


class SchemaLearner:
    """Extract recurring ordered event patterns from episodes."""

    def __init__(self, min_support: int = 2) -> None:
        if min_support < 2:
            raise ValueError("min_support must be >= 2")
        self.min_support = min_support
        self._counts: dict[tuple[str, tuple[str, ...]], int] = defaultdict(int)
        self._context_totals: dict[str, int] = defaultdict(int)

    def observe(self, context: str, steps: tuple[str, ...] | list[str]) -> None:
        context = str(context).strip()
        normalized = tuple(str(step).strip() for step in steps if str(step).strip())
        if not context or not normalized:
            raise ValueError("context and steps must be non-empty")
        self._counts[(context, normalized)] += 1
        self._context_totals[context] += 1

    def schemas(self) -> list[Schema]:
        output: list[Schema] = []
        for (context, steps), support in self._counts.items():
            if support < self.min_support:
                continue
            total = self._context_totals[context]
            confidence = support / total if total else 0.0
            name = f"{context}:{'->'.join(steps)}"
            output.append(Schema(name, context, steps, support, confidence))
        return sorted(output, key=lambda s: (-s.confidence, -s.support, s.name))

    def predict(self, context: str) -> Schema | None:
        candidates = [s for s in self.schemas() if s.context == context]
        return candidates[0] if candidates else None
