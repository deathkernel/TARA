from src.polyglot.archive import CandidateArchive
from src.polyglot.benchmark import BenchmarkResult
from src.polyglot.candidate import PolyglotCandidate
from src.polyglot.novelty import NoveltyAnalyzer, similarity, structural_signature


def result(source: str, verified: bool = True) -> BenchmarkResult:
    candidate = PolyglotCandidate(problem="sorting", language="python", source=source)
    return BenchmarkResult(candidate, 2 if verified else 1, 2, 1.0 if verified else 0.5, 2.0, () if verified else ("case 2 failed",))


def test_python_signature_ignores_identifier_names():
    first = PolyglotCandidate("sorting", "python", "def solve(x):\n    y = sorted(x)\n    return y\n")
    second = PolyglotCandidate("sorting", "python", "def process(data):\n    result = sorted(data)\n    return result\n")
    assert structural_signature(first) == structural_signature(second)
    assert similarity(first, second) == 1.0


def test_analyzer_marks_exact_duplicate(tmp_path):
    archive = CandidateArchive(tmp_path / "archive.jsonl")
    candidate = PolyglotCandidate("sorting", "python", "print('ok')\n")
    archive.save(result(candidate.source))
    report = NoveltyAnalyzer(archive).analyze(candidate)
    assert report.exact_match is True
    assert report.new_to_archive is False
    assert report.nearest[0].similarity == 1.0


def test_analyzer_marks_structurally_new_candidate(tmp_path):
    archive = CandidateArchive(tmp_path / "archive.jsonl")
    archive.save(result("def solve(x):\n    return sorted(x)\n"))
    candidate = PolyglotCandidate("sorting", "python", "class X:\n    pass\n")
    report = NoveltyAnalyzer(archive, threshold=0.95).analyze(candidate)
    assert report.exact_match is False
    assert report.new_to_archive is True
