from src.tara_mind import TaraMind
from src.tara_mind.cognition.appraisal import AppraisalInput
from src.tara_mind.cognition.attention import AttentionItem
from src.tara_mind.cognition.executive import ActionCandidate


def test_unified_mind_connects_core_modules():
    mind = TaraMind()
    mind.set_goal("g", "reach B")
    first = mind.process(
        text="move from A to B",
        attention_items=[AttentionItem("goal", goal_relevance=1.0)],
        appraisal_input=AppraisalInput(
            goal_relevance=1.0,
            goal_congruence=0.5,
            controllability=0.8,
            certainty=0.8,
        ),
        actions=[ActionCandidate("move", "g", expected_value=1.0, confidence=0.8)],
        transition=("A", "B"),
    )
    assert first.attended[0].item.content == "goal"
    assert first.decision.action_id == "move"
    assert first.world_update.observation.next_state == "B"
    assert first.reflection is not None
    assert 0.0 <= first.metacognition.confidence <= 1.0

    second = mind.process(
        text="move from A to C",
        attention_items=[AttentionItem("new", novelty=1.0)],
        appraisal_input=AppraisalInput(goal_relevance=1.0, novelty=1.0, certainty=0.2),
        transition=("A", "C"),
    )
    assert second.world_update is not None
    assert second.reflection is not None
