import pytest

from src.agent import AgentLoop
from src.integration import TARAEngine


def test_agent_runs_verified_plan_to_completion():
    agent = AgentLoop(TARAEngine())
    agent.engine.set_goal("finish", ("done",))
    agent.engine.plan(["prepare", "finish"])

    assert agent.next_task().kind == "task"
    agent.submit_result("ok", "ok")
    assert agent.next_task().task == "finish"
    agent.submit_result(42, 42)
    assert agent.complete
    assert agent.next_task().kind == "complete"


def test_failed_result_does_not_advance_plan():
    agent = AgentLoop(TARAEngine())
    agent.engine.set_goal("finish")
    agent.engine.plan(["attempt", "final"])
    agent.submit_result("bad", "good")
    assert not agent.complete
    assert agent.engine.state.plan.current.description == "attempt"
    assert agent.history[-1].kind == "failure"


def test_agent_can_replan_after_failure():
    agent = AgentLoop(TARAEngine())
    agent.engine.set_goal("finish")
    agent.engine.plan(["attempt", "old-final"])
    agent.submit_result("bad", "good")
    event = agent.recover(["alternative", "new-final"])
    assert event.kind == "replan"
    assert agent.next_task().task == "alternative"
    agent.submit_result("done", "done")
    agent.submit_result("final", "final")
    assert agent.complete


def test_agent_requires_active_plan():
    agent = AgentLoop(TARAEngine())
    with pytest.raises(ValueError):
        agent.next_task()
    with pytest.raises(ValueError):
        agent.submit_result("x", "x")
    with pytest.raises(ValueError):
        agent.recover(["replacement"])
