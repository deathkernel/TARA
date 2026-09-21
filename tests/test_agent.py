from src.agent import AgentLoop
from src.integration import TARAEngine
from src.memory import LongTermMemory


def _planned_agent():
    engine = TARAEngine()
    engine.set_goal("finish task", ["done"])
    engine.plan(["step one", "step two"])
    return AgentLoop(engine=engine)


def test_agent_observe_remember_and_recall():
    memory = LongTermMemory()
    agent = _planned_agent()
    agent.memory = memory
    event = agent.observe("Python notes", remember_key="language")
    assert event.kind == "observe"
    recalled = agent.recall("language")
    assert recalled.kind == "recall"
    assert recalled.value[0]["value"] == "Python notes"
    assert memory.metadata("language")["access_count"] == 1


def test_agent_advances_only_after_verification():
    agent = _planned_agent()
    assert agent.next_task().task == "step one"
    failure = agent.submit_result("wrong", "right")
    assert failure.kind == "failure"
    assert agent.engine.state.plan.current_index == 0
    success = agent.submit_result("right", "right")
    assert success.kind == "advance"
    assert agent.engine.state.plan.current_index == 1


def test_agent_can_replan_after_failure():
    agent = _planned_agent()
    agent.submit_result("wrong", "right")
    event = agent.recover(["replacement"])
    assert event.kind == "replan"
    assert event.task == "replacement"


def test_agent_reports_completion():
    agent = _planned_agent()
    agent.submit_result("done", "done")
    agent.submit_result("done", "done")
    assert agent.complete
    assert agent.next_task().kind == "complete"
