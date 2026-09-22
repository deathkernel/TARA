"""Perception primitives for TARA.

Perception remains separate from action: this module converts observable input
into bounded structured representations. TemporalPerception adds a canonical
confidence-aware stream for downstream reasoning.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .temporal_perception import NormalizedObservation, ObservationNormalizer, TemporalContext, TemporalPerception


@dataclass(frozen=True)
class DocumentState:
    """A compact representation of a text-like document."""

    path: str
    size_bytes: int
    suffix: str
    text: str


@dataclass(frozen=True)
class SystemState:
    """Explicit system facts supplied by a trusted observer."""

    facts: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_mapping(cls, facts: Mapping[str, object]):
        if not isinstance(facts, Mapping):
            raise TypeError("facts must be a mapping")
        normalized = []
        for key in sorted(facts):
            if not isinstance(key, str) or not key.strip():
                raise ValueError("system-state keys must be non-empty strings")
            normalized.append((key, str(facts[key])))
        return cls(tuple(normalized))

    def as_dict(self):
        return dict(self.facts)


def perceive_document(path, max_bytes=1_000_000):
    """Read a bounded UTF-8 text document into a structured state."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(str(source))
    size = source.stat().st_size
    if size > max_bytes:
        raise ValueError("document exceeds max_bytes")
    text = source.read_text(encoding="utf-8")
    return DocumentState(str(source), size, source.suffix.lower(), text)


def perceive_system_state(facts):
    """Normalize externally observed system facts without performing actions."""
    return SystemState.from_mapping(facts)


def summarize_screen_state(elements):
    """Create a deterministic text summary from observed screen elements."""
    if elements is None:
        raise ValueError("elements must not be None")
    summaries = []
    for element in elements:
        if not isinstance(element, Mapping):
            raise TypeError("each screen element must be a mapping")
        kind = str(element.get("type", "element"))
        label = str(element.get("label", "")).strip()
        text = str(element.get("text", "")).strip()
        detail = " ".join(part for part in (label, text) if part)
        summaries.append(f"{kind}: {detail}" if detail else kind)
    return "\n".join(summaries)


def normalize_observation(value, **kwargs) -> NormalizedObservation:
    """Convert one adapter payload into TARA's canonical observation type."""
    return ObservationNormalizer().normalize(value, **kwargs)


def build_temporal_context(observations, *, limit=16, min_confidence=0.0) -> TemporalContext:
    """Order and bound observations for reasoning."""
    engine = TemporalPerception(max_history=max(limit, 1))
    engine.ingest_many(observations)
    return engine.context(limit=limit, min_confidence=min_confidence)
