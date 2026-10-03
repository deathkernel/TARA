from src.tara_mind.cognition.appraisal import AppraisalEngine, AppraisalInput


def test_goal_congruence_increases_valence():
    engine = AppraisalEngine()
    positive = engine.evaluate(AppraisalInput(goal_congruence=1.0, controllability=1.0, certainty=1.0))
    negative = engine.evaluate(AppraisalInput(goal_congruence=-1.0, controllability=1.0, certainty=1.0))
    assert positive.valence > negative.valence


def test_uncertainty_and_novelty_increase_arousal_proxy():
    engine = AppraisalEngine()
    calm = engine.evaluate(AppraisalInput(novelty=0.0, certainty=1.0))
    alert = engine.evaluate(AppraisalInput(novelty=1.0, certainty=0.0))
    assert alert.arousal > calm.arousal


def test_regulation_damps_extremes():
    state = AppraisalEngine().evaluate(AppraisalInput(goal_relevance=1.0, goal_congruence=1.0, novelty=1.0))
    regulated = AppraisalEngine.regulate(state)
    assert regulated.arousal < state.arousal
