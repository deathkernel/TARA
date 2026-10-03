"""Prediction-outcome reflection and error monitoring for TARA Baby.

Grounding: cognitive control benefits from monitoring outcomes against goals and
adjusting behavior when conflict or error is detected. The mechanism here is an
explicit computational reflection trace, not subjective introspection.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reflection:
    matched: bool
    deviation: float
    revise_belief: bool
    replan: bool
    reason: str


class ReflectionEngine:
    """Compare expected and observed outcomes and trigger corrective control."""

    def assess(
        self,
        expected: float,
        observed: float,
        tolerance: float = 0.1,
        confidence: float = 0.5,
    ) -> Reflection:
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        deviation = abs(float(observed) - float(expected))
        matched = deviation <= tolerance
        revise = (not matched) and confidence >= 0.4
        replan = deviation > max(tolerance, 0.2)
        reason = "outcome matched prediction" if matched else "prediction-outcome mismatch requires review"
        return Reflection(matched, deviation, revise, replan, reason)

    def compare_states(
        self,
        expected: str | None,
        observed: str,
        confidence: float = 0.5,
    ) -> Reflection:
        """Compare symbolic world states for the cognitive loop."""
        if not observed:
            raise ValueError("observed state must be non-empty")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if expected is None:
            return Reflection(False, 1.0, False, True, "no prediction was available")
        matched = expected == observed
        return Reflection(
            matched=matched,
            deviation=0.0 if matched else 1.0,
            revise_belief=not matched and confidence >= 0.4,
            replan=not matched,
            reason="state matched prediction" if matched else "state mismatch requires replanning",
        )
