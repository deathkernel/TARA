from src.cognitive_memory import CognitiveMemory
from src.learning_consolidation import LearningConsolidator


def test_consolidates_promoted_checkpoint(tmp_path):
    memory = CognitiveMemory(tmp_path / "tara.db")
    report = LearningConsolidator(memory).consolidate(
        checkpoint="candidate.pt",
        dataset_fingerprint="abc123",
        validation_loss=1.25,
        regression_passed=True,
        promoted=True,
        outcome="accepted",
        metrics={"relative_improvement": 0.12},
    )

    assert report.promoted is True
    assert len(report.memory_ids) == 2
    assert memory.search(kind="learning_cycle", limit=10)
    checkpoints = memory.search(kind="model_checkpoint", verification="verified", limit=10)
    assert checkpoints[0]["content"]["checkpoint"] == "candidate.pt"
    assert memory.related(report.memory_ids[0], relation="produced_checkpoint")[0]["id"] == report.memory_ids[1]


def test_rejected_result_is_stored_without_checkpoint_promotion(tmp_path):
    memory = CognitiveMemory(tmp_path / "tara.db")
    report = LearningConsolidator(memory).consolidate(
        checkpoint="candidate.pt",
        dataset_fingerprint="abc123",
        validation_loss=2.0,
        regression_passed=False,
        promoted=False,
        outcome="rejected_regression",
    )

    assert report.memory_ids and len(report.memory_ids) == 1
    assert memory.search(kind="model_checkpoint", limit=10) == []
    record = memory.get(report.memory_ids[0])
    assert record["verification"] == "unverified"
    assert record["content"]["promoted"] is False
