from src.tara_mind.cognition.attention import AttentionItem, AttentionWeights, SelectiveAttention

def test_goal_relevance_gets_priority():
    controller = SelectiveAttention()
    selected = controller.allocate([
        AttentionItem("noise", goal_relevance=0.0),
        AttentionItem("task", goal_relevance=1.0),
    ], budget=1)
    assert selected[0].item.content == "task"

def test_budget_is_bounded():
    controller = SelectiveAttention()
    items = [AttentionItem(str(i), salience=i / 5.0) for i in range(6)]
    selected = controller.allocate(items, budget=3)
    assert len(selected) == 3
    assert selected[0].score >= selected[1].score >= selected[2].score

def test_allocations_normalize():
    controller = SelectiveAttention()
    selected = controller.allocate([
        AttentionItem("a", goal_relevance=1.0),
        AttentionItem("b", novelty=1.0),
        AttentionItem("c", urgency=1.0),
    ], budget=3)
    assert abs(sum(item.score for item in selected) - 1.0) < 1e-9

def test_invalid_weights_are_rejected():
    try:
        AttentionWeights(goal_relevance=0.5)
    except ValueError:
        return
    raise AssertionError("invalid weights should be rejected")
