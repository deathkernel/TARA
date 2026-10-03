from src.sandbox_runner import run_candidate, validate_candidate


def test_valid_candidate_runs():
    result = run_candidate("def solve(value):\n    return sorted(value)", [3, 1, 2])
    assert not result.ok
    assert "disabled" in result.error


def test_forbidden_import_is_rejected():
    ok, reason = validate_candidate("import os\ndef solve(value):\n    return value")
    assert not ok
    assert "import not allowed" in reason


def test_wrong_output_is_still_executable_but_not_correct():
    result = run_candidate("def solve(value):\n    return value", [2, 1])
    assert not result.ok
    assert "disabled" in result.error


def test_timeout_is_reported():
    result = run_candidate("def solve(value):\n    while True:\n        pass", [], timeout=0.1)
    assert not result.ok
    assert "disabled" in result.error
