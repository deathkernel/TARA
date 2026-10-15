import json
from pathlib import Path

from src.teacher_distillation import generate_teacher_dataset
from training_preflight import run_preflight


class FakeProvider:
    model = "fake-teacher-v1"

    def complete(self, prompt, **kwargs):
        return "A sufficiently detailed candidate answer that can be independently reviewed: " + prompt


def _approved_teacher_dataset(path: Path) -> None:
    generate_teacher_dataset(FakeProvider(), path, per_domain=1)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["review_status"] = "approved"
        row["review"] = {"decision": "approve", "reviewer": "test-reviewer", "notes": "independently checked"}
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_preflight_accepts_human_approved_teacher_dataset(tmp_path: Path):
    dataset = tmp_path / "approved_teacher.jsonl"
    _approved_teacher_dataset(dataset)
    report = run_preflight(dataset, context=64)
    assert report["checks"]
    assert report["ready"] is True
    assert any(item["name"] == "teacher_approval" and item["passed"] for item in report["checks"])
    assert any(item["name"] == "dataset_audit" and item["passed"] for item in report["checks"])


def test_preflight_blocks_unapproved_dataset_by_default(tmp_path: Path):
    dataset = tmp_path / "unapproved_legacy_style.jsonl"
    dataset.write_text(json.dumps({"problem": "legacy-style sample", "solution": "not reviewed"}) + "\\n", encoding="utf-8")
    report = run_preflight(dataset, context=64)
    assert report["ready"] is False
    approval = next(item for item in report["checks"] if item["name"] == "teacher_approval")
    assert approval["passed"] is False


def test_preflight_rejects_missing_dataset(tmp_path: Path):
    report = run_preflight(tmp_path / "missing.jsonl")
    assert report["ready"] is False
    assert report["checks"][0]["name"] == "dataset_exists"
