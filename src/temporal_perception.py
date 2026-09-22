"""Advanced perception and temporal-context primitives for TARA.

The module turns heterogeneous observations into deterministic, confidence-aware
records and assembles bounded temporal context for downstream reasoning. It is
intentionally model-agnostic: learned perception adapters can be plugged in
later without changing the temporal contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class NormalizedObservation:
    """Canonical observation emitted by a perception adapter."""

    source: str
    kind: str
    content: str
    confidence: float = 1.0
    timestamp: float = 0.0
    attributes: tuple[tuple[str, str], ...] = ()
    observation_id: str = ""

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if not self.kind.strip():
            raise ValueError("kind must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if not self.observation_id:
            payload = {
                "source": self.source,
                "kind": self.kind,
                "content": self.content,
                "confidence": self.confidence,
                "timestamp": self.timestamp,
                "attributes": self.attributes,
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            object.__setattr__(self, "observation_id", digest)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.observation_id,
            "source": self.source,
            "kind": self.kind,
            "content": self.content,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class TemporalRelation:
    """Pairwise temporal relation between two observations."""

    earlier_id: str
    later_id: str
    delta_seconds: float


@dataclass(frozen=True)
class TemporalContext:
    """Bounded context window ordered by observation time."""

    observations: tuple[NormalizedObservation, ...]
    relations: tuple[TemporalRelation, ...] = ()

    def high_confidence(self, threshold: float = 0.7) -> tuple[NormalizedObservation, ...]:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        return tuple(item for item in self.observations if item.confidence >= threshold)

    def as_prompt_context(self, max_chars: int = 12_000) -> str:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        lines = ["Temporal observations (oldest first):"]
        for item in self.observations:
            lines.append(
                f"- [{item.timestamp:.3f}] {item.kind} from {item.source} "
                f"(confidence={item.confidence:.2f}): {item.content}"
            )
        if self.relations:
            lines.append("Temporal relations:")
            for relation in self.relations:
                lines.append(
                    f"- {relation.earlier_id[:10]} -> {relation.later_id[:10]} "
                    f"(+{relation.delta_seconds:.3f}s)"
                )
        text = "\n".join(lines)
        return text[:max_chars]


class ObservationNormalizer:
    """Normalize mappings, strings, and adapter payloads into observations."""

    def normalize(
        self,
        value: Any,
        *,
        source: str = "unknown",
        kind: str = "observation",
        timestamp: float | None = None,
        confidence: float | None = None,
    ) -> NormalizedObservation:
        if isinstance(value, NormalizedObservation):
            return value
        if not source.strip():
            raise ValueError("source must not be empty")
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).timestamp()

        attributes: dict[str, str] = {}
        if isinstance(value, Mapping):
            content = value.get("content", value.get("text", value.get("value", "")))
            source = str(value.get("source", source))
            kind = str(value.get("kind", value.get("type", kind)))
            if confidence is None:
                confidence = float(value.get("confidence", 1.0))
            timestamp = float(value.get("timestamp", timestamp))
            raw_attributes = value.get("attributes", {})
            if isinstance(raw_attributes, Mapping):
                attributes = {str(k): str(v) for k, v in raw_attributes.items()}
            else:
                raise TypeError("attributes must be a mapping")
        else:
            content = value

        if confidence is None:
            confidence = 1.0
        if isinstance(content, (dict, list, tuple)):
            content = json.dumps(content, sort_keys=True, default=str)
        else:
            content = str(content)
        return NormalizedObservation(
            source=str(source).strip(),
            kind=str(kind).strip(),
            content=content.strip(),
            confidence=float(confidence),
            timestamp=float(timestamp),
            attributes=tuple(sorted(attributes.items())),
        )


class TemporalPerception:
    """Ingest, order, deduplicate and contextualize observations."""

    def __init__(self, *, max_history: int = 256, normalizer: ObservationNormalizer | None = None):
        if max_history <= 0:
            raise ValueError("max_history must be positive")
        self.max_history = max_history
        self.normalizer = normalizer or ObservationNormalizer()
        self._history: list[NormalizedObservation] = []
        self._seen: set[str] = set()

    def ingest(self, value: Any, **kwargs: Any) -> NormalizedObservation:
        observation = self.normalizer.normalize(value, **kwargs)
        if observation.observation_id not in self._seen:
            self._history.append(observation)
            self._seen.add(observation.observation_id)
            self._trim()
        return observation

    def ingest_many(self, values: Iterable[Any], **kwargs: Any) -> tuple[NormalizedObservation, ...]:
        return tuple(self.ingest(value, **kwargs) for value in values)

    def ordered(self, *, newest_first: bool = False) -> tuple[NormalizedObservation, ...]:
        ordered = sorted(self._history, key=lambda item: (item.timestamp, item.observation_id))
        if newest_first:
            ordered.reverse()
        return tuple(ordered)

    def context(self, *, limit: int = 16, min_confidence: float = 0.0) -> TemporalContext:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        selected = [item for item in self.ordered() if item.confidence >= min_confidence]
        selected = selected[-limit:]
        relations = tuple(
            TemporalRelation(a.observation_id, b.observation_id, b.timestamp - a.timestamp)
            for a, b in zip(selected, selected[1:])
        )
        return TemporalContext(tuple(selected), relations)

    def clear(self) -> None:
        self._history.clear()
        self._seen.clear()

    def _trim(self) -> None:
        if len(self._history) > self.max_history:
            self._history.sort(key=lambda item: (item.timestamp, item.observation_id))
            removed = self._history[:-self.max_history]
            self._history = self._history[-self.max_history:]
            for item in removed:
                self._seen.discard(item.observation_id)


def temporal_prompt_context(
    observations: Sequence[NormalizedObservation], *, limit: int = 16, max_chars: int = 12_000
) -> str:
    """Build bounded reasoning context without mutating perception state."""
    engine = TemporalPerception(max_history=max(len(observations), 1))
    engine.ingest_many(observations)
    return engine.context(limit=limit).as_prompt_context(max_chars=max_chars)
