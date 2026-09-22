import json

from src.capability_suite import capability_benchmark, capability_cases
from src.training_experiment import TrainingExperiment
from src.training_pipeline import TrainingConfig


def test_capability_suite_has_multiple_independent_categories():
    cases = capability_cases()
    categories = {case.category for case in cases}
    assert len(cases) >= 20
    assert {"coding", "reasoning", "memory", "planning", "tool_use", "algorithms", "learning"} <= categories


def test_capability_benchmark_is_solvable_and_deterministic():
    answers = {
        "Compute 17 * 3": 51,
        "Reverse the string 'tara'": "arat",
        "Sort 3, 1, 2 ascending": "1 2 3",
        "A is taller than B and B is taller than C. Is A taller than C?": True,
        "If all birds are animals, are all animals birds?": False,
        "What is 12 / 3?": 4,
        "Recall token TARA-37": "TARA-37",
        "Recall that the project name is TARA": "TARA",
        "Recall sequence: perceive, reason, plan": "perceive reason plan",
        "Before executing a task, what should happen first?": "plan",
        "Task B depends on Task A. Which runs first?": "A",
        "Should an unvalidated plan be executed immediately?": False,
        "Need to read a local file. Which capability is required?": "file-read",
        "After a tool reports success, what should happen before trusting the result?": "verify",
        "If the emergency stop is active, should a new tool action execute?": False,
        "What property must an algorithm satisfy before promotion?": "correctness",
        "What should be compared when optimizing two verified algorithms?": "performance",
        "If a candidate is faster but fails a required test, promote it?": False,
        "What is replay used to reduce during continual learning?": "forgetting",
        "Should unverified knowledge enter the verified learning set?": False,
        "What should gate a claimed learning improvement?": "evidence",
    }
    benchmark = capability_benchmark()
    first = benchmark.run(lambda prompt: answers[prompt])
    second = benchmark.run(lambda prompt: answers[prompt])
    assert first.passed
    assert first.overall_score == 1.0
    assert first.fingerprint == second.fingerprint


def test_training_experiment_blocks_split_overlap(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"text": "short"}) + "\n", encoding="utf-8")
    experiment = TrainingExperiment(TrainingConfig(steps=1, context=2, batch_size=1, embedding_dim=4, ff_dim=8, heads=2))
    report = experiment.audit(data)
    assert report.records == 1
    assert report.issues


def test_training_experiment_manifest_schema(tmp_path, monkeypatch):
    data = tmp_path / "data.jsonl"
    rows = [{"problem": f"sort list {i}", "solution": "return sorted(value)", "tests": "3,1,2"} for i in range(8)]
    data.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    class FakeSummary:
        dataset_fingerprint = "d" * 64
        checkpoint = str(tmp_path / "model.pt")
        metrics_path = str(tmp_path / "metrics.jsonl")
        start_step = 0
        final_step = 1
        train_loss = 1.2
        validation_loss = 1.4
        stopped_early = False
        device = "cpu"

    monkeypatch.setattr("src.training_experiment.TrainingPipeline.train", lambda *a, **k: FakeSummary())
    manifest = tmp_path / "manifest.json"
    report = TrainingExperiment(TrainingConfig(steps=1)).run(data, tmp_path / "model.pt", manifest_path=manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == report.experiment_id
    assert payload["audit_fingerprint"]
    assert payload["dataset_fingerprint"] == "d" * 64
