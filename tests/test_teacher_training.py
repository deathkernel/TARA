import json
from pathlib import Path

import pytest

from src.teacher_distillation import generate_teacher_dataset
from src.teacher_training import train_curated_candidate, validate_approved_dataset


class FakeProvider:
    model = "fake-teacher-v1"

    def complete(self, prompt, **kwargs):
        return "A candidate answer with enough detail: " + prompt


def make_dataset(path: Path, *, approved: bool = True):
    generate_teacher_dataset(FakeProvider(), path, per_domain=1)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["review_status"] = "approved" if approved else "unverified_candidate"
        if approved:
            row["review"] = {"decision": "approve", "reviewer": "reviewer-1", "notes": "checked"}
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return rows


class FakeTrainingReport:
    def as_dict(self):
        return {"experiment_id": "test-experiment", "final_step": 2}


class FakeTrainer:
    def __init__(self, config, device=None):
        assert config.steps == 2

    def run(self, dataset, output):
        assert validate_approved_dataset(dataset)["records"] == 9
        Path(output).write_bytes(b"fake candidate checkpoint")
        return FakeTrainingReport()


class FakeBenchmark:
    name = "tara-heldout-capability-v1"
    fingerprint = "fixed-suite"

    def __init__(self, score):
        self.overall_score = score
        self.categories = [type("Category", (), {"category": "reasoning", "score": score})()]

    def as_dict(self):
        return {"name": self.name, "fingerprint": self.fingerprint, "overall_score": self.overall_score,
                "categories": [{"category": item.category, "score": item.score} for item in self.categories]}


class FakeEvaluation:
    def __init__(self, checkpoint, score):
        self.checkpoint = str(checkpoint)
        self.benchmark = FakeBenchmark(score)
        self.fingerprint = "eval-" + str(score)

    def as_dict(self):
        return {"checkpoint": self.checkpoint, "fingerprint": self.fingerprint,
                "benchmark": self.benchmark.as_dict()}


class FakeDecision:
    def __init__(self, accepted):
        self.accepted = accepted
        self.fingerprint = "decision-fingerprint"

    def as_dict(self):
        return {"accepted": self.accepted, "reason": "test decision", "fingerprint": self.fingerprint}


class FakeEvaluator:
    def __init__(self, accepted=True):
        self.accepted = accepted

    def evaluate(self, checkpoint, *, name):
        return FakeEvaluation(checkpoint, 0.8 if "baseline" in str(checkpoint) else 0.9)

    def compare(self, baseline, candidate):
        return FakeDecision(self.accepted)


def test_training_refuses_unreviewed_teacher_data(tmp_path: Path):
    dataset = tmp_path / "teacher.jsonl"
    make_dataset(dataset, approved=False)
    with pytest.raises(ValueError, match="only review_status=approved"):
        validate_approved_dataset(dataset)


def test_training_requires_reviewer_metadata(tmp_path: Path):
    dataset = tmp_path / "teacher.jsonl"
    rows = make_dataset(dataset)
    rows[0]["review"]["reviewer"] = ""
    dataset.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewer"):
        validate_approved_dataset(dataset)


def test_trains_separate_candidate_and_writes_evidence_report(tmp_path: Path):
    dataset = tmp_path / "approved.jsonl"
    make_dataset(dataset)
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"baseline")
    candidate, report = tmp_path / "candidate.pt", tmp_path / "evidence.json"
    result = train_curated_candidate(dataset, baseline, candidate, report, steps=2,
        trainer_factory=FakeTrainer, evaluator=FakeEvaluator())
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert candidate.is_file()
    assert result["decision"]["accepted"] is True
    assert payload["dataset"]["records"] == 9
    assert payload["training"]["experiment_id"] == "test-experiment"
    assert payload["category_deltas"][0]["delta"] > 0
    assert payload["report_fingerprint"]


def test_training_never_overwrites_existing_candidate_or_report(tmp_path: Path):
    dataset = tmp_path / "approved.jsonl"
    make_dataset(dataset)
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"baseline")
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"preserve me")
    with pytest.raises(FileExistsError, match="candidate checkpoint"):
        train_curated_candidate(dataset, baseline, candidate, tmp_path / "report.json",
            steps=2, trainer_factory=FakeTrainer, evaluator=FakeEvaluator())
    assert candidate.read_bytes() == b"preserve me"


def test_nonaccepted_candidate_is_reported_without_promotion(tmp_path: Path):
    dataset = tmp_path / "approved.jsonl"
    make_dataset(dataset)
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"baseline")
    candidate, report = tmp_path / "candidate.pt", tmp_path / "report.json"
    result = train_curated_candidate(dataset, baseline, candidate, report, steps=2,
        trainer_factory=FakeTrainer, evaluator=FakeEvaluator(accepted=False))
    assert result["decision"]["accepted"] is False
    assert candidate.exists() and report.exists()
    assert "No checkpoint is promoted automatically" in result["interpretation"]


def test_training_rejects_artifact_path_collisions(tmp_path: Path):
    dataset = tmp_path / "approved.jsonl"
    make_dataset(dataset)
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"baseline")
    with pytest.raises(ValueError, match="paths must be distinct"):
        train_curated_candidate(dataset, baseline, dataset, tmp_path / "report.json",
            steps=2, trainer_factory=FakeTrainer, evaluator=FakeEvaluator())
