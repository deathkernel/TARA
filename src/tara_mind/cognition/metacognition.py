"""Metacognitive monitoring for TARA Baby.

Grounding: metacognition involves self-evaluation of cognitive performance,
including confidence and uncertainty. The implementation is an explicit
computational monitor layered over primary cognition; it is not subjective
self-awareness.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetacognitiveReport:
    confidence: float
    uncertainty: float
    should_verify: bool
    reason: str


class MetacognitiveMonitor:
    """Estimate confidence from evidence, ambiguity and observed error."""

    def assess(
        self,
        base_confidence: float,
        evidence_count: int = 0,
        prediction_error: float = 0.0,
        ambiguity: float = 0.0,
        consequence_cost: float = 0.0,
    ) -> MetacognitiveReport:
        if not 0.0 <= base_confidence <= 1.0:
            raise ValueError("base_confidence must be in [0, 1]")
        if evidence_count < 0:
            raise ValueError("evidence_count must be >= 0")
        for name, value in (("ambiguity", ambiguity), ("consequence_cost", consequence_cost)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if prediction_error < 0:
            raise ValueError("prediction_error must be >= 0")

        evidence_factor = 1.0 - 1.0 / (1.0 + evidence_count)
        error_factor = 1.0 / (1.0 + prediction_error)
        confidence = (
            0.45 * base_confidence
            + 0.25 * evidence_factor
            + 0.30 * error_factor
        )
        confidence *= 1.0 - 0.55 * ambiguity
        confidence = min(1.0, max(0.0, confidence))
        uncertainty = 1.0 - confidence
        verify = (
            uncertainty >= 0.45
            or prediction_error >= 1.0
            or consequence_cost * uncertainty >= 0.20
        )
        reason = "verification recommended" if verify else "confidence sufficient for current evidence"
        return MetacognitiveReport(confidence, uncertainty, verify, reason)

    @staticmethod
    def compare_revision(old_confidence: float, new_confidence: float) -> float:
        """Return signed confidence change for an explicit self-update trace."""
        if not 0.0 <= old_confidence <= 1.0 or not 0.0 <= new_confidence <= 1.0:
            raise ValueError("confidence values must be in [0, 1]")
        return float(new_confidence) - float(old_confidence)
