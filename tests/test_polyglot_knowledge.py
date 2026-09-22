import json

from src.polyglot.benchmark import BenchmarkResult
from src.polyglot.candidate import PolyglotCandidate
from src.polyglot.knowledge import KnowledgeExtractor


def make_result(source="def solve(x):\n    return sorted(x)\n", verified=True):
    candidate = PolyglotCandidate("sorting", "python", source)
    return BenchmarkResult(
        candidate=candidate,
        passed=4 if verified else 2,
        total=4,
        correctness=1.0 if verified else 0.5,
        total_runtime_ms=2.5,
        failures=() if verified else ("case 3 failed",),
    )


def test_extractor_accepts_verified_candidates_only():
    extractor = KnowledgeExtractor()
    records = extractor.extract([make_result(), make_result("bad", verified=False)])
    assert len(records) == 1
    assert records[0].verified is True
    assert records[0].language == "python"
    assert "INPUT=" in records[0].tests


def test_extractor_deduplicates_by_source_fingerprint():
    extractor = KnowledgeExtractor()
    records = extractor.extract([make_result(), make_result()])
    assert len(records) == 1


def test_export_is_valid_jsonl_and_keeps_training_contract(tmp_path):
    extractor = KnowledgeExtractor()
    records = extractor.extract([make_result()])
    path = tmp_path / "verified.jsonl"
    assert extractor.export_jsonl(records, path) == 1
    item = json.loads(path.read_text(encoding="utf-8"))
    assert item["problem"] == "sorting"
    assert item["solution"].startswith("def solve")
    assert item["tests"]
    assert item["verified"] is True
