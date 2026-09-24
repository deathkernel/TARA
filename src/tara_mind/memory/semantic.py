"""Semantic memory for generalized, source-traceable knowledge.

Scientific grounding: semantic memory stores generalized knowledge rather than
one particular event. TARA keeps provenance and confidence so consolidation can
change a belief without erasing the originating evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SemanticFact:
    key: str
    value: str
    confidence: float = 0.5
    evidence_count: int = 1
    sources: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.confidence = float(self.confidence)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.evidence_count < 1:
            raise ValueError("evidence_count must be >= 1")


class SemanticMemory:
    """Maintain consolidated facts with evidence provenance."""

    def __init__(self) -> None:
        self._facts: dict[str, SemanticFact] = {}

    def learn(self, key: str, value: str, confidence: float = 0.5, source: str | None = None) -> SemanticFact:
        if not key or not value:
            raise ValueError("key and value must be non-empty")
        confidence = min(1.0, max(0.0, float(confidence)))
        existing = self._facts.get(key)
        if existing is None:
            fact = SemanticFact(key, value, confidence, 1, {source} if source else set())
            self._facts[key] = fact
            return fact

        if existing.value == value:
            existing.confidence = min(1.0, 1.0 - (1.0 - existing.confidence) * (1.0 - confidence))
            existing.evidence_count += 1
            if source:
                existing.sources.add(source)
            return existing

        # Contradictory evidence lowers confidence instead of silently overwriting history.
        existing.confidence *= 0.5
        existing.evidence_count += 1
        if source:
            existing.sources.add(source)
        return existing

    def get(self, key: str) -> SemanticFact | None:
        return self._facts.get(key)

    def search(self, query: str, limit: int = 5) -> list[SemanticFact]:
        q = query.lower().strip()
        if not q:
            return []
        if limit < 0:
            raise ValueError("limit must be >= 0")
        tokens = set(q.split())
        ranked = sorted(
            self._facts.values(),
            key=lambda fact: (len(tokens & set((fact.key + " " + fact.value).lower().split())), fact.confidence),
            reverse=True,
        )
        return ranked[:limit]

    def __len__(self) -> int:
        return len(self._facts)
