from pathlib import Path

import pytest

import src.self_test as self_test
from src.self_test import TestRun, run_test_suite, summarize_pytest_output


def test_test_run_passed_depends_only_on_return_code():
    assert TestRun(("python", "-m", "pytest"), 0, "1 passed", "").passed
    assert not TestRun(("python", "-m", "pytest"), 1, "1 failed", "").passed


def test_pytest_output_summary_extracts_counts():
    summary = summarize_pytest_output("10 passed, 2 failed, 1 skipped in 0.2s")
    assert summary["passed"] == 10
    assert summary["failed"] == 2
    assert summary["skipped"] == 1
    assert summary["errors"] == 0


def test_pytest_output_summary_handles_stderr():
    summary = summarize_pytest_output("", "1 passed, 1 error in 0.1s")
    assert summary["passed"] == 1
    assert summary["errors"] == 1
    assert "1 error" in summary["output"]


def test_runner_rejects_invalid_root_and_timeout(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_test_suite(tmp_path / "missing")
    with pytest.raises(ValueError):
        run_test_suite(tmp_path, timeout=0)


def test_runner_uses_fixed_pytest_command(monkeypatch, tmp_path):
    captured = {}

    class Completed:
        returncode = 0
        stdout = "3 passed in 0.1s"
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr(self_test.subprocess, "run", fake_run)
    result = run_test_suite(tmp_path)
    assert result.passed
    assert result.command[-3:] == ("-m", "pytest", "-q")
    assert captured["kwargs"]["cwd"] == Path(tmp_path)
    assert captured["kwargs"]["check"] is False
