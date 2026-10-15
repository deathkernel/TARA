import csv, json
from pathlib import Path
import pytest
from src.teacher_dataset_audit import audit_teacher_dataset
from src.teacher_distillation import DOMAIN_TASKS, generate_teacher_dataset
from src.teacher_review import export_review_queue, import_review_decisions
from src.teacher_heldout_eval import evaluate_heldout

class FakeProvider:
    model = "fake-teacher-v1"
    def complete(self, prompt, **kwargs):
        return f"Careful candidate answer with enough detail: {prompt}"

def _dataset(path: Path, per_domain: int = 1):
    generate_teacher_dataset(FakeProvider(), path, per_domain=per_domain)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

def test_audit_accepts_generated_dataset_and_reports_all_domains(tmp_path: Path):
    dataset = tmp_path / "teacher.jsonl"; _dataset(dataset)
    report = audit_teacher_dataset(dataset)
    assert report.passed and report.valid_records == len(DOMAIN_TASKS)
    assert set(report.domain_counts) == set(DOMAIN_TASKS)
    assert all(count == 1 for count in report.domain_counts.values()) and report.fingerprint

def test_audit_flags_tampered_fingerprint_and_malformed_json(tmp_path: Path):
    dataset = tmp_path / "teacher.jsonl"; rows = _dataset(dataset)
    rows[0]["fingerprint"] = "not-the-real-fingerprint"
    dataset.write_text(json.dumps(rows[0]) + "\n{not-json}\n", encoding="utf-8")
    report = audit_teacher_dataset(dataset)
    assert not report.passed
    assert {issue.code for issue in report.issues} >= {"fingerprint-mismatch", "malformed-json"}

def test_review_export_import_only_emits_explicitly_approved_rows(tmp_path: Path):
    dataset = tmp_path / "teacher.jsonl"; rows = _dataset(dataset)
    queue, curated = tmp_path / "review.csv", tmp_path / "approved.jsonl"
    assert export_review_queue(dataset, queue) == len(rows)
    with queue.open("r", encoding="utf-8", newline="") as stream: items = list(csv.DictReader(stream))
    for i, item in enumerate(items):
        item["decision"] = "approve" if i == 0 else "reject"; item["reviewer"] = "reviewer-1"
        item["notes"] = "independently checked" if i == 0 else "not reliable"
    with queue.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=items[0].keys()); writer.writeheader(); writer.writerows(items)
    counts = import_review_decisions(dataset, queue, curated)
    approved = [json.loads(line) for line in curated.read_text(encoding="utf-8").splitlines()]
    assert counts == {"approved": 1, "rejected": len(rows)-1, "total_candidates": len(rows)}
    assert len(approved) == 1 and approved[0]["review_status"] == "approved"
    assert approved[0]["review"]["reviewer"] == "reviewer-1"

def test_review_import_rejects_incomplete_decisions_and_preserves_inputs(tmp_path: Path):
    dataset = tmp_path / "teacher.jsonl"; _dataset(dataset); queue = tmp_path / "review.csv"
    export_review_queue(dataset, queue); before = dataset.read_bytes()
    with pytest.raises(ValueError, match="review incomplete"):
        import_review_decisions(dataset, queue, tmp_path / "approved.jsonl")
    assert dataset.read_bytes() == before and not (tmp_path / "approved.jsonl").exists()

def test_review_import_rejects_changed_answer(tmp_path: Path):
    dataset = tmp_path / "teacher.jsonl"; _dataset(dataset); queue = tmp_path / "review.csv"
    export_review_queue(dataset, queue)
    with queue.open("r", encoding="utf-8", newline="") as stream: items = list(csv.DictReader(stream))
    for item in items: item["decision"], item["reviewer"] = "approve", "reviewer-1"
    items[0]["answer"] += " edited after export"
    with queue.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=items[0].keys()); writer.writeheader(); writer.writerows(items)
    with pytest.raises(ValueError, match="content changed"):
        import_review_decisions(dataset, queue, tmp_path / "approved.jsonl")

class _Benchmark:
    name = "tara-capability-v1"; fingerprint = "fixed-suite-fingerprint"
    def __init__(self, score):
        self.overall_score = score
        self.categories = [type("Category", (), {"category": "reasoning", "score": score})()]
    def as_dict(self):
        return {"name": self.name, "fingerprint": self.fingerprint, "overall_score": self.overall_score,
                "categories": [{"category": c.category, "score": c.score} for c in self.categories]}
class _Evaluation:
    def __init__(self, checkpoint, score):
        self.checkpoint = str(checkpoint); self.benchmark = _Benchmark(score)
        self.fingerprint = f"eval-{score}"; self.model_fingerprint = f"model-{score}"
    def as_dict(self):
        return {"checkpoint": self.checkpoint, "benchmark": self.benchmark.as_dict(),
                "fingerprint": self.fingerprint, "model_fingerprint": self.model_fingerprint}
class _Decision:
    def __init__(self, accepted, baseline, candidate):
        self.accepted = accepted; self.reason = "improved" if accepted else "did not improve"
        self.baseline_score = baseline; self.candidate_score = candidate
        self.fingerprint = f"decision-{baseline}-{candidate}"; self.regression = not accepted
    def as_dict(self): return vars(self)
class _Evaluator:
    def evaluate(self, checkpoint, *, name=None):
        return _Evaluation(checkpoint, 0.4 if str(checkpoint).endswith("baseline.pt") else 0.6)
    def compare(self, baseline, candidate):
        return _Decision(candidate.benchmark.overall_score > baseline.benchmark.overall_score,
                         baseline.benchmark.overall_score, candidate.benchmark.overall_score)

def test_heldout_eval_persists_comparison_and_never_overwrites(tmp_path: Path):
    baseline, candidate, report = tmp_path / "baseline.pt", tmp_path / "candidate.pt", tmp_path / "heldout.json"
    baseline.write_bytes(b"baseline"); candidate.write_bytes(b"candidate")
    result = evaluate_heldout(baseline, candidate, report, evaluator=_Evaluator())
    assert result["decision"]["accepted"] is True
    assert result["category_deltas"][0]["delta"] == pytest.approx(0.2)
    assert json.loads(report.read_text(encoding="utf-8"))["benchmark_fingerprint"] == "fixed-suite-fingerprint"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        evaluate_heldout(baseline, candidate, report, evaluator=_Evaluator())
