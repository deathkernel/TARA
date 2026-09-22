from src.continual_learning_engine import ContinualLearningEngine


def test_verified_balanced_replay_is_deduplicated_and_bounded():
    records = [
        {"text": "a", "domain": "code", "importance": 2, "verified": True},
        {"text": "a", "domain": "code", "importance": 2, "verified": True},
        {"text": "b", "domain": "math", "importance": 1, "verified": True},
        {"text": "c", "domain": "security", "importance": 3, "verified": False},
    ]
    batch = ContinualLearningEngine(seed=7, replay_size=2).build_replay(records)
    assert len(batch.examples) == 2
    assert {x.domain for x in batch.examples} == {"code", "math"}
    assert sum(batch.weights) > 0


def test_promotion_gate_rejects_forgetting():
    result = ContinualLearningEngine().evaluate([1.0, 1.0, 1.0], [1.0, 0.9, 1.0])
    assert not result.accepted
    assert "forgetting" in result.reason


def test_promotion_accepts_improvement_without_forgetting():
    result = ContinualLearningEngine().evaluate([0.8, 0.8], [0.9, 0.85])
    assert result.accepted
    assert result.metrics.improvement_score > 0
