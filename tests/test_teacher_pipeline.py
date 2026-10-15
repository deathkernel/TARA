import csv
import json
from pathlib import Path

import pytest

from src.teacher_pipeline import curate_pipeline, prepare_pipeline, train_pipeline


class FakeProvider:
    model = "fake-teacher-v1"

    def complete(self, prompt, **kwargs):
        return "A sufficiently detailed candidate response: " + prompt


def _prepare(tmp_path: Path):
    dataset = tmp_path / "candidates.jsonl"
    audit = tmp_path / "audit.json"
    queue = tmp_path / "review.csv"
    manifest = tmp_path / "prepare-manifest.json"
    result = prepare_pipeline(FakeProvider(), dataset, audit, queue, manifest, per_domain=1)
    return dataset, audit, queue, manifest, result


def _complete_review(queue: Path, *, incomplete: bool = False):
    with queue.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for index, row in enumerate(rows):
        row["decision"] = "" if incomplete and index == 0 else ("approve" if index == 0 else "reject")
        row["reviewer"] = "" if incomplete and index == 0 else "reviewer-1"
        row["notes"] = "independently checked" if row["decision"] == "approve" else "reviewed"
    with queue.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_generates_audits_and_exports_review_queue(tmp_path: Path):
    dataset, audit, queue, manifest, result = _prepare(tmp_path)
    assert dataset.is_file() and audit.is_file() and queue.is_file() and manifest.is_file()
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    assert audit_payload["passed"] is True
    assert result["stage"] == "awaiting_human_review"
    assert result["generation_counts"]["total"] == 9
    assert len(list(csv.DictReader(queue.open(encoding="utf-8", newline="")))) == 9


def test_prepare_refuses_to_overwrite_existing_artifacts(tmp_path: Path):
    dataset, audit, queue, manifest, _ = _prepare(tmp_path)
    original = dataset.read_bytes()
    with pytest.raises(FileExistsError, match="candidate dataset"):
        prepare_pipeline(FakeProvider(), dataset, tmp_path / "new-audit.json",
                         tmp_path / "new-review.csv", tmp_path / "new-manifest.json", per_domain=1)
    assert dataset.read_bytes() == original


def test_curate_imports_only_approved_rows_and_records_handoff(tmp_path: Path):
    dataset, _, queue, _, _ = _prepare(tmp_path)
    _complete_review(queue)
    approved, manifest = tmp_path / "approved.jsonl", tmp_path / "curation-manifest.json"
    result = curate_pipeline(dataset, queue, approved, manifest)
    rows = [json.loads(line) for line in approved.read_text(encoding="utf-8").splitlines()]
    assert result["stage"] == "approved_dataset_ready"
    assert result["review_counts"] == {"approved": 1, "rejected": 8, "total_candidates": 9}
    assert len(rows) == 1
    assert rows[0]["review_status"] == "approved"
    assert rows[0]["review"]["reviewer"] == "reviewer-1"
    assert manifest.is_file()


def test_curate_incomplete_review_does_not_create_approved_dataset(tmp_path: Path):
    dataset, _, queue, _, _ = _prepare(tmp_path)
    _complete_review(queue, incomplete=True)
    approved, manifest = tmp_path / "approved.jsonl", tmp_path / "curation-manifest.json"
    with pytest.raises(ValueError, match="review incomplete"):
        curate_pipeline(dataset, queue, approved, manifest)
    assert not approved.exists()
    assert not manifest.exists()


def test_prepare_rejects_colliding_output_paths(tmp_path: Path):
    same = tmp_path / "same.json"
    with pytest.raises(ValueError, match="must be distinct"):
        prepare_pipeline(FakeProvider(), same, same, tmp_path / "review.csv",
                         tmp_path / "manifest.json", per_domain=1)


def test_train_stage_delegates_to_evidence_gated_training(monkeypatch, tmp_path: Path):
    import src.teacher_pipeline as pipeline
    expected = {"decision": {"accepted": True}, "report_fingerprint": "abc"}
    observed = {}

    def fake_train(dataset, baseline, candidate, report, **kwargs):
        observed.update(dataset=dataset, baseline=baseline, candidate=candidate, report=report, **kwargs)
        return expected

    monkeypatch.setattr(pipeline, "train_curated_candidate", fake_train)
    result = train_pipeline("approved.jsonl", "baseline.pt", "candidate.pt", "evidence.json",
                            steps=7, device="cpu", max_new_tokens=9)
    assert result is expected
    assert observed == {
        "dataset": "approved.jsonl", "baseline": "baseline.pt", "candidate": "candidate.pt",
        "report": "evidence.json", "steps": 7, "device": "cpu", "max_new_tokens": 9,
    }
