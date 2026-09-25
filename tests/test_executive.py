from src.tara_mind.cognition.executive import ActionCandidate, ExecutiveController, Goal


def test_active_goal_is_highest_priority():
    controller = ExecutiveController()
    controller.add_goal(Goal("low", "low", 0.2))
    controller.add_goal(Goal("high", "high", 0.9))
    assert controller.active_goal().goal_id == "high"


def test_best_positive_action_is_selected():
    controller = ExecutiveController()
    controller.add_goal(Goal("g", "goal", 1.0))
    decision = controller.choose([
        ActionCandidate("bad", "g", expected_value=0.1, expected_cost=0.5),
        ActionCandidate("good", "g", expected_value=0.8, confidence=0.5),
    ])
    assert decision.action_id == "good"
    assert not decision.blocked


def test_risky_action_is_inhibited():
    controller = ExecutiveController(risk_tolerance=0.2)
    controller.add_goal(Goal("g", "goal", 1.0))
    decision = controller.choose([ActionCandidate("danger", "g", expected_value=1.0, risk=0.8)])
    assert decision.blocked


def test_confirmation_is_not_executed():
    controller = ExecutiveController()
    controller.add_goal(Goal("g", "goal", 1.0))
    decision = controller.choose([ActionCandidate("tool", "g", expected_value=1.0, requires_confirmation=True)])
    assert decision.blocked
