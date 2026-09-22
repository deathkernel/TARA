"""Structured, inspectable reasoning state for TARA.

This layer does not pretend to expose private chain-of-thought. It stores
compact, auditable reasoning artifacts: facts, assumptions, hypotheses,
evidence, contradictions and verification outcomes. The representation is
model-agnostic and can be populated by a future learned reasoner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping


class ClaimKind(str, Enum):
    FACT = "fact"
    ASSUMPTION = "assumption"
    HYPOTHESIS = "hypothesis"


@dataclass(frozen=True)
class Claim:
    id: str
    text: str
    kind: ClaimKind
    confidence: float = 1.0
    source: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("claim id must not be empty")
        if not self.text.strip():
            raise ValueError("claim text must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class Evidence:
    id: str
    claim_id: str
    content: str
    supports: bool
    strength: float = 1.0
    source: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.claim_id.strip():
            raise ValueError("evidence and claim ids must not be empty")
        if not self.content.strip():
            raise ValueError("evidence content must not be empty")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("evidence strength must be between 0 and 1")


@dataclass(frozen=True)
class Contradiction:
    left_claim_id: str
    right_claim_id: str
    reason: str
    severity: float = 1.0

    def __post_init__(self) -> None:
        if self.left_claim_id == self.right_claim_id:
            raise ValueError("a claim cannot contradict itself")
        if not self.reason.strip():
            raise ValueError("contradiction reason must not be empty")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must be between 0 and 1")


@dataclass(frozen=True)
class VerificationResult:
    verified: bool
    score: float
    checked_claim_ids: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("verification score must be between 0 and 1")


@dataclass
class ReasoningState:
    """Bounded structured state for one reasoning problem."""

    goal: str
    facts: dict[str, Claim] = field(default_factory=dict)
    assumptions: dict[str, Claim] = field(default_factory=dict)
    hypotheses: dict[str, Claim] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    verification: VerificationResult | None = None

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ValueError("reasoning goal must not be empty")

    @property
    def claims(self) -> tuple[Claim, ...]:
        return tuple(self.facts.values()) + tuple(self.assumptions.values()) + tuple(self.hypotheses.values())

    def add_claim(self, claim: Claim) -> Claim:
        target = {
            ClaimKind.FACT: self.facts,
            ClaimKind.ASSUMPTION: self.assumptions,
            ClaimKind.HYPOTHESIS: self.hypotheses,
        }[claim.kind]
        target[claim.id] = claim
        return claim

    def add_evidence(self, evidence: Evidence) -> Evidence:
        if not any(claim.id == evidence.claim_id for claim in self.claims):
            raise KeyError(f"unknown claim: {evidence.claim_id}")
        self.evidence.append(evidence)
        return evidence

    def add_contradiction(self, contradiction: Contradiction) -> Contradiction:
        known = {claim.id for claim in self.claims}
        if contradiction.left_claim_id not in known or contradiction.right_claim_id not in known:
            raise KeyError("contradiction references an unknown claim")
        if contradiction not in self.contradictions:
            self.contradictions.append(contradiction)
        return contradiction

    def snapshot(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "facts": [claim.__dict__.copy() for claim in self.facts.values()],
            "assumptions": [claim.__dict__.copy() for claim in self.assumptions.values()],
            "hypotheses": [claim.__dict__.copy() for claim in self.hypotheses.values()],
            "evidence": [item.__dict__.copy() for item in self.evidence],
            "contradictions": [item.__dict__.copy() for item in self.contradictions],
            "verification": None if self.verification is None else self.verification.__dict__.copy(),
        }


class HypothesisGenerator:
    """Generate deterministic baseline hypotheses from facts and a goal."""

    def generate(self, state: ReasoningState, candidates: Iterable[str]) -> tuple[Claim, ...]:
        generated = []
        for index, text in enumerate(candidates):
            normalized = str(text).strip()
            if not normalized:
                continue
            digest = hashlib.sha256(
                f"{state.goal}\n{normalized}".encode("utf-8")
            ).hexdigest()[:16]
            claim = Claim(f"h-{digest}-{index}", normalized, ClaimKind.HYPOTHESIS, 0.5)
            state.add_claim(claim)
            generated.append(claim)
        return tuple(generated)


class EvidenceTracker:
    """Attach evidence and recompute hypothesis confidence conservatively."""

    def attach(self, state: ReasoningState, evidence: Evidence) -> Evidence:
        state.add_evidence(evidence)
        claim = next(item for item in state.claims if item.id == evidence.claim_id)
        supports = [item for item in state.evidence if item.claim_id == claim.id]
        support = sum(item.strength for item in supports if item.supports)
        opposition = sum(item.strength for item in supports if not item.supports)
        score = max(0.0, min(1.0, 0.5 + 0.25 * (support - opposition)))
        updated = Claim(claim.id, claim.text, claim.kind, score, claim.source)
        target = {ClaimKind.FACT: state.facts, ClaimKind.ASSUMPTION: state.assumptions, ClaimKind.HYPOTHESIS: state.hypotheses}[claim.kind]
        target[claim.id] = updated
        return evidence


class ContradictionDetector:
    """Detect explicit conflicts and optionally use a caller-supplied predicate."""

    def detect(
        self,
        state: ReasoningState,
        predicate: Callable[[Claim, Claim], bool] | None = None,
    ) -> tuple[Contradiction, ...]:
        predicate = predicate or self._default_conflict
        claims = state.claims
        found: list[Contradiction] = []
        for index, left in enumerate(claims):
            for right in claims[index + 1 :]:
                if predicate(left, right):
                    contradiction = Contradiction(left.id, right.id, "claims are explicitly incompatible")
                    state.add_contradiction(contradiction)
                    found.append(contradiction)
        return tuple(found)

    @staticmethod
    def _default_conflict(left: Claim, right: Claim) -> bool:
        a = left.text.strip().lower()
        b = right.text.strip().lower()
        if a == b:
            return False
        pairs = (("true", "false"), ("yes", "no"), ("enabled", "disabled"), ("online", "offline"))
        return any((x in a and y in b) or (y in a and x in b) for x, y in pairs)


class ReasoningVerifier:
    """Verify a reasoning state with explicit, deterministic checks."""

    def verify(self, state: ReasoningState, *, required_claim_ids: Iterable[str] = ()) -> VerificationResult:
        required = tuple(required_claim_ids)
        known = {claim.id for claim in state.claims}
        failures: list[str] = []
        for claim_id in required:
            if claim_id not in known:
                failures.append(f"missing claim: {claim_id}")
        for contradiction in state.contradictions:
            if contradiction.severity > 0.8:
                failures.append(
                    f"high-severity contradiction: {contradiction.left_claim_id}/{contradiction.right_claim_id}"
                )
        evidence_score = 1.0
        if state.hypotheses:
            evidence_score = sum(claim.confidence for claim in state.hypotheses.values()) / len(state.hypotheses)
        contradiction_penalty = min(0.5, sum(item.severity for item in state.contradictions) * 0.1)
        score = max(0.0, min(1.0, evidence_score - contradiction_penalty))
        result = VerificationResult(not failures and score >= 0.5, score, tuple(required), tuple(failures))
        state.verification = result
        return result


class StructuredReasoner:
    """High-level facade for an inspectable reasoning cycle."""

    def __init__(self, *, max_evidence: int = 128, max_contradictions: int = 64):
        if max_evidence <= 0 or max_contradictions <= 0:
            raise ValueError("reasoning bounds must be positive")
        self.max_evidence = max_evidence
        self.max_contradictions = max_contradictions
        self.hypotheses = HypothesisGenerator()
        self.evidence = EvidenceTracker()
        self.contradictions = ContradictionDetector()
        self.verifier = ReasoningVerifier()

    def start(self, goal: str) -> ReasoningState:
        return ReasoningState(goal)

    def reason(
        self,
        state: ReasoningState,
        *,
        facts: Iterable[str] = (),
        assumptions: Iterable[str] = (),
        hypotheses: Iterable[str] = (),
        evidence: Iterable[tuple[str, str, bool, float]] = (),
        required_claim_ids: Iterable[str] = (),
    ) -> ReasoningState:
        for index, text in enumerate(facts):
            state.add_claim(Claim(f"f-{index}", str(text), ClaimKind.FACT, 1.0))
        for index, text in enumerate(assumptions):
            state.add_claim(Claim(f"a-{index}", str(text), ClaimKind.ASSUMPTION, 0.5))
        self.hypotheses.generate(state, hypotheses)
        for index, (claim_id, content, supports, strength) in enumerate(evidence):
            self.evidence.attach(state, Evidence(f"e-{index}", claim_id, content, supports, strength))
        self.contradictions.detect(state)
        if len(state.evidence) > self.max_evidence:
            del state.evidence[:-self.max_evidence]
        if len(state.contradictions) > self.max_contradictions:
            del state.contradictions[:-self.max_contradictions]
        self.verifier.verify(state, required_claim_ids=required_claim_ids)
        return state

    @staticmethod
    def to_json(state: ReasoningState) -> str:
        return json.dumps(state.snapshot(), sort_keys=True, default=lambda value: value.value if isinstance(value, Enum) else str(value))
