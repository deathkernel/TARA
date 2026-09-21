import pytest

from src.integration import TARAEngine


def test_engine_requires_goal_before_planning():
    engine = TARAEngine()
    with pytest.raises(ValueError):
        engine.plan(["step"])


def test_engine_connects_goal_plan_observation_and_results():
    engine = TARAEngine(working_memory_capacity=2)
    goal = engine.set_goal("finish task", ("done",))
    plan = engine.plan(["prepare", "execute"])
    engine.observe("ready")
    engine.observe("running")
    engine.observe("finished")
    engine.record_result({"done": True})

    assert goal.description == "finish task"
    assert plan.current.description == "prepare"
    assert engine.state.observations.recent() == ["running", "finished"]
    assert engine.snapshot() == {
        "goal": "finish task",
        "plan": ["prepare", "execute"],
        "current_step": 0,
        "observations": ["running", "finished"],
        "results": [{"done": True}],
    }


def test_setting_new_goal_resets_plan_and_results():
    engine = TARAEngine()
    engine.set_goal("old")
    engine.plan(["old step"])
    engine.record_result("old result")
    engine.set_goal("new")
    assert engine.state.plan is None
    assert engine.state.results == []
    assert engine.state.goal.description == "new"
