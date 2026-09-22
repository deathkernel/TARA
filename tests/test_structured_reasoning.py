from src.structured_reasoning import (
    Claim,
    ClaimKind,
    Contradiction,
    Evidence,
    ReasoningVerifier,
    StructuredReasoner,
)


def test_reasoning_separates_facts_assumptions_and_hypotheses():
    reasoner = StructuredReasoner()
    state = reasoner.start("determine system state")
    reasoner.reason(
        state,
        facts=["network is online"],
        assumptions=["sensor is reliable"],
        hypotheses=["service is reachable"],
    )
    assert len(state.facts) == 1
    assert len(state.assumptions) == 1
    assert len(state.hypotheses) == 1


def test_evidence_updates_hypothesis_confidence_and_verification():
    reasoner = StructuredReasoner()
    state = reasoner.start("choose a valid hypothesis")
    reasoner.reason(
        state,
        hypotheses=["candidate works"],
    )
    hypothesis_id = next(iter(state.hypotheses))
    reasoner.reason(
        state,
        evidence=[(hypothesis_id, "benchmark passed", True, 1.0)],
        required_claim_ids=[hypothesis_id],
    )
    assert state.hypotheses[hypothesis_id].confidence > 0.5
    assert state.verification is not None
    assert state.verification.verified


def test_contradiction_detection_is_explicit_and_verifiable():
    reasoner = StructuredReasoner()
    state = reasoner.start("resolve system status")
    state.add_claim(Claim("a", "system is online", ClaimKind.FACT))
    state.add_claim(Claim("b", "system is offline", ClaimKind.FACT))
    reasoner.contradictions.detect(state)
    result = ReasoningVerifier().verify(state)
    assert len(state.contradictions) == 1
    assert not result.verified
