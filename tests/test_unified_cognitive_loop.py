from src.experience_learning import Experience, ExperienceLearningEngine
from src.unified_cognitive_loop import ActionOutcome, UnifiedCognitiveLoop


class FakeMemory:
    def __init__(self):
        self.items = []

    def retrieve(self, query, limit=8):
        return tuple(self.items[-limit:])

    def remember(self, key, value, importance=1.0):
        self.items.append((key, value, importance))


def test_unified_loop_runs_perception_to_learning():
    from src.goal_progress import GoalProgress
    loop = UnifiedCognitiveLoop(memory=FakeMemory(), progress=GoalProgress(), max_cycles=4)
    report = loop.run(
        "build feature",
        ["design", "verify"],
        ["input one", "input two"],
        action_executor=lambda request: ActionOutcome(request.step.description, True, score=1.0),
        expected=lambda value: isinstance(value, str),
    )
    assert report.completed
    assert len(report.traces) == 2
    assert report.traces[0].learning.updates
    assert report.traces[0].reflection.experience.success
    assert report.fingerprint


def test_failed_verification_generates_learning_and_stops_progress():
    loop = UnifiedCognitiveLoop(memory=FakeMemory())
    report = loop.run(
        "goal",
        ["step"],
        ["perception"],
        action_executor=lambda request: ActionOutcome("wrong", True, score=-1.0, feedback="mismatch"),
        expected="right",
    )
    assert not report.completed
    assert report.traces[0].verified is False
    assert report.traces[0].learning.signals
    assert report.traces[0].progress.status == "in-progress"


def test_tool_selection_is_optional_but_wires_into_cycle():
    from src.tool_intelligence import ToolCapability
    from src.goal_progress import GoalProgress
    from src.tool_intelligence import ToolIntelligence
    tools = ToolIntelligence()
    tools.register(ToolCapability("calculator", "math", ("compute",), reliability=1.0))
    loop = UnifiedCognitiveLoop(memory=FakeMemory(), tools=tools, progress=GoalProgress())
    trace = loop.start("compute", ["calculate"])
    item = loop.cycle(
        "2+2",
        action_executor=lambda request: ActionOutcome(4, True, score=1.0),
        expected=4,
        required_capabilities=("compute",),
    )
    assert item.selected_tool == "calculator"
