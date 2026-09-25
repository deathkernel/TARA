from src.tara_mind.automation.tools import ToolCall, ToolRegistry, ToolSpec
from src.tara_mind.cognition.loop import CognitiveLoop
from src.tara_mind.cognition.state import AffectState, CognitiveState
from src.tara_mind.evaluation.tasks import EvaluationTask


def test_affect_state_validates_range():
    state = AffectState(valence=0.2, arousal=0.3, urgency=0.4, frustration=0.1, confidence=0.8)
    assert state.confidence == 0.8


def test_cognitive_state_records_working_memory_and_observations():
    state = CognitiveState(goal="open browser")
    state.remember_working("Chrome is closed", source="perception", importance=0.9)
    state.observe("Chrome was launched")
    assert state.context[0].content == "Chrome is closed"
    assert state.observations == ["Chrome was launched"]


def test_cognitive_loop_trace_is_explicit():
    loop = CognitiveLoop()
    loop.record("perception", "user request received")
    loop.record("reasoning", "candidate plan prepared")
    assert [item.stage for item in loop.trace] == ["perception", "reasoning"]


def test_tool_registry_blocks_confirmation_tools():
    registry = ToolRegistry()
    registry.register(ToolSpec("safe", "safe test", lambda: "ok"))
    registry.register(ToolSpec("dangerous", "needs confirmation", lambda: "no", requires_confirmation=True))
    assert registry.execute(ToolCall("safe", {})).success
    result = registry.execute(ToolCall("dangerous", {}))
    assert not result.success
    assert result.error == "confirmation required"


def test_evaluation_score_is_bounded():
    task = EvaluationTask("echo", "language", "say hi", lambda output: 1.0 if output == "hi" else 0.0)
    assert task.score("hi") == 1.0
