from src.tara_mind.cognition.planning import ModelBasedPlanner
from src.tara_mind.cognition.predictive_map import PredictiveMap


def test_planner_finds_multistep_path():
    model = PredictiveMap()
    model.observe_transition("A", "B")
    model.observe_transition("B", "C")
    planner = ModelBasedPlanner(model)
    plan = planner.plan("A", "C")
    assert plan is not None
    assert plan.states == ("A", "B", "C")


def test_planner_returns_none_without_route():
    model = PredictiveMap()
    model.observe_transition("A", "B")
    assert ModelBasedPlanner(model).plan("A", "Z") is None
