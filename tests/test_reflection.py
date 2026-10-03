from src.tara_mind.cognition.reflection import ReflectionEngine


def test_matching_outcome_needs_no_replan():
    result = ReflectionEngine().assess(1.0, 1.05, tolerance=0.1)
    assert result.matched
    assert not result.replan


def test_large_prediction_error_triggers_replan():
    result = ReflectionEngine().assess(1.0, 2.0, tolerance=0.1, confidence=0.9)
    assert not result.matched
    assert result.revise_belief
    assert result.replan
