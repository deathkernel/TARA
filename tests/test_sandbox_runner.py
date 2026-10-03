from src.sandbox_runner import run_candidate, validate_candidate


def test_valid_candidate_is_validated_but_not_executed_on_host():
    source = "def solve(value):\n    return sorted(value)"
    ok, reason = validate_candidate(source)
    assert ok
    assert reason == ""

    result = run_candidate(source, [3, 1, 2])
    assert not result.ok
    assert "isolated execution backend" in result.error
    assert result.output is None


def test_forbidden_import_is_rejected():
    ok, reason = validate_candidate("import os\ndef solve(value):\n    return value")
    assert not ok
    assert "import not allowed" in reason


def test_invalid_candidate_never_reaches_execution_boundary():
    result = run_candidate("def solve(value):\n    return value", [2, 1])
    assert result.ok is False
    assert "isolated execution backend" in result.error


def test_timeout_argument_is_validated_even_when_execution_is_disabled():
    try:
        run_candidate("def solve(value):\n    while True:\n        pass", [], timeout=0)
    except ValueError as exc:
        assert "timeout" in str(exc)
    else:
        raise AssertionError("invalid timeout should fail")
