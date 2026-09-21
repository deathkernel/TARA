import pytest

from src.reasoning import Goal, Plan, Task, decompose_goal, make_plan, replan, verify_goal, verify_step


def test_goal_requires_description():
    with pytest.raises(ValueError):
        Goal("")


def test_goal_and_task_are_structured():
    goal = Goal("learn Python", ("learned",))
    task = Task("study functions", goal.description)
    assert goal.description == "learn Python"
    assert task.parent == "learn Python"


def test_decompose_goal_preserves_order():
    goal = Goal("learn Python")
    tasks = decompose_goal(goal, ["read", "practice", "review"])
    assert [task.description for task in tasks] == ["read", "practice", "review"]
    assert all(task.parent == goal.description for task in tasks)


def test_plan_tracks_current_step_and_completion():
    plan = make_plan(Goal("learn"), ["read", "practice"])
    assert plan.current.description == "read"
    assert not plan.complete
    plan.advance()
    assert plan.current.description == "practice"
    plan.advance()
    assert plan.complete
    assert plan.current is None


def test_plan_rejects_invalid_current_index():
    with pytest.raises(ValueError):
        Plan(Goal("x"), [Task("a")], current_index=2)


def test_verification_is_explicit_and_deterministic():
    assert verify_step("done", "done")
    assert not verify_step("done", "failed")
    goal = Goal("finish", ("built", "tested"))
    assert verify_goal(goal, {"built": True, "tested": True})
    assert not verify_goal(goal, {"built": True, "tested": False})


def test_replan_replaces_failed_step_and_future_steps():
    goal = Goal("finish")
    plan = make_plan(goal, ["first", "bad", "old-final"])
    plan.current_index = 1
    updated = replan(plan, 1, ["alternative", "new-final"])
    assert [task.description for task in updated.steps] == ["first", "alternative", "new-final"]
    assert updated.current_index == 1
    assert updated.current.description == "alternative"


def test_replan_rejects_invalid_inputs():
    plan = make_plan(Goal("finish"), ["one"])
    with pytest.raises(IndexError):
        replan(plan, 2, ["replacement"])
    with pytest.raises(ValueError):
        replan(plan, 0, [])
