from src.polyglot.candidate import PolyglotCandidate
from src.polyglot.execution import PolyglotExecutor


def test_python_candidate_executes():
    candidate = PolyglotCandidate(
        problem="echo",
        language="python",
        source="print(input().upper())",
    )
    result = PolyglotExecutor(run_timeout=2).execute(candidate, "tara\n")
    assert result.success
    assert result.stdout.strip() == "TARA"


def test_unknown_language_is_rejected():
    candidate = PolyglotCandidate(problem="x", language="brainfuck", source="+")
    try:
        PolyglotExecutor().execute(candidate)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown language should be rejected")


def test_timeout_is_reported():
    candidate = PolyglotCandidate(
        problem="loop",
        language="python",
        source="while True: pass",
    )
    result = PolyglotExecutor(run_timeout=0.2).execute(candidate)
    assert result.timed_out
    assert not result.success
