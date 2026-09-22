from src.polyglot.archive import CandidateArchive
from src.polyglot.benchmark import BenchmarkResult
from src.polyglot.candidate import PolyglotCandidate
from src.polyglot.learning import VerifiedKnowledgeLearner


def make_result(source: str, verified: bool = True) -> BenchmarkResult:
    candidate = PolyglotCandidate("sorting", "python", source)
    return BenchmarkResult(
        candidate=candidate,
        passed=4 if verified else 3,
        total=4,
        correctness=1.0 if verified else 0.75,
        total_runtime_ms=2.0,
        failures=() if verified else ("case 4 failed",),
    )


def test_learning_export_contains_only_verified_candidates(tmp_path):
    archive = CandidateArchive(tmp_path / "archive.jsonl")
    archive.save(make_result("print('verified')"))
    archive.save(make_result("print('failed')", verified=False))

    output = tmp_path / "training.jsonl"
    exported = VerifiedKnowledgeLearner(archive).export(output)

    assert exported.records == 1
    line = output.read_text(encoding="utf-8").strip()
    assert '"verified": true' in line
    assert "print('verified')" in line
    assert "print('failed')" not in line


def test_learning_export_can_scope_to_problem(tmp_path):
    archive = CandidateArchive(tmp_path / "archive.jsonl")
    archive.save(make_result("print('sorting')"))

    output = tmp_path / "training.jsonl"
    exported = VerifiedKnowledgeLearner(archive).export(output, problem="binary_search")

    assert exported.records == 0
    assert output.read_text(encoding="utf-8") == ""
