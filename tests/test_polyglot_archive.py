from src.polyglot.archive import CandidateArchive
from src.polyglot.benchmark import BenchmarkResult
from src.polyglot.candidate import PolyglotCandidate


def make_result(source="print('ok')"):
    candidate = PolyglotCandidate(problem="sorting", language="python", source=source)
    return BenchmarkResult(candidate, 2, 2, 1.0, 3.5, ())


def test_archive_deduplicates_candidates(tmp_path):
    archive = CandidateArchive(tmp_path / "candidates.jsonl")
    result = make_result()
    assert archive.save(result) is True
    assert archive.save(result) is False
    assert archive.stats("sorting") == {"candidates": 1, "verified": 1, "languages": 1}


def test_archive_best_prefers_correctness_then_runtime(tmp_path):
    archive = CandidateArchive(tmp_path / "candidates.jsonl")
    slow = make_result("print('slow')")
    fast = BenchmarkResult(
        PolyglotCandidate(problem="sorting", language="python", source="print('fast')"),
        1, 2, 0.5, 0.1, ("case 2: failed",),
    )
    archive.save(slow)
    archive.save(fast)
    assert archive.best("sorting")["benchmark"]["verified"] is True
    assert archive.history("sorting")[0]["candidate"]["source"] == "print('slow')"
