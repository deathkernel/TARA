from src.polyglot.candidate import PolyglotCandidate
from src.polyglot.execution import PolyglotExecutor


def test_python_candidate_is_rejected_at_disabled_execution_boundary():
    candidate = PolyglotCandidate(
        problem="echo",
        language="python",
        source="print(input().upper())",
    )
    result = PolyglotExecutor(run_timeout=2).execute(candidate, "tara\n")
    assert not result.success
    assert result.phase == "disabled"
    assert "disabled" in result.error


def test_unknown_language_is_rejected():
    candidate = PolyglotCandidate(problem="x", language="brainfuck", source="+")
    try:
        PolyglotExecutor().execute(candidate)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown language should be rejected")


def test_generated_code_never_times_out_because_host_execution_is_disabled():
    candidate = PolyglotCandidate(
        problem="loop",
        language="python",
        source="while True: pass",
    )
    result = PolyglotExecutor(run_timeout=0.2).execute(candidate)
    assert not result.timed_out
    assert not result.success
    assert result.phase == "disabled"
