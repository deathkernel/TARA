from src.emotional_intelligence import EmotionalIntelligence


def test_detects_hinglish_sadness_without_extra_model():
    engine = EmotionalIntelligence()
    assessment = engine.assess("Bhai aaj bahut udaas hoon aur rona aa raha hai")

    assert assessment.label == "sadness"
    assert assessment.intensity > 0
    assert assessment.evidence
    assert "gentle" in assessment.response_guidance.lower()


def test_detects_frustration_and_gives_actionable_guidance():
    assessment = EmotionalIntelligence().assess("TARA samajh nahi aa raha, not working")

    assert assessment.label == "frustration"
    assert "actionable" in assessment.response_guidance


def test_neutral_message_stays_neutral():
    assessment = EmotionalIntelligence().assess("What is the capital of France?")

    assert assessment.label == "neutral"
    assert assessment.intensity == 0.0
    assert assessment.evidence == ()


def test_system_guidance_does_not_claim_real_tara_feelings():
    engine = EmotionalIntelligence()
    guidance = engine.system_guidance(engine.assess("I feel sad"))

    assert "never claim TARA literally feels emotions" in guidance
    assert "not a diagnosis" in guidance


def test_companion_guidance_uses_only_preferred_nicknames_and_respects_distress():
    engine = EmotionalIntelligence()
    guidance = engine.system_guidance(engine.assess("Bhai, aaj bahut udaas hoon"))

    assert "balanced romantic companion" in guidance
    assert "Baby, jaan, and babu" in guidance
    assert "Never call the user beta, baccha, child, son, or daughter" in guidance
    assert "prioritize empathy over flirting or teasing" in guidance
    assert "not a claim of literal human feelings" in guidance
