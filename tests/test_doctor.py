import pytest
from src.doctor import main, scan_repository, scaffold_command, suggest_commands

def test_scan_finds_silent_exception_and_marker(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "sample.py").write_text("try:\n work()\nexcept Exception:\n pass  # FIXME inspect this\n", encoding="utf-8")
    codes = {f.code for f in scan_repository(tmp_path).findings}
    assert "SILENT_BROAD_EXCEPTION" in codes
    assert "REVIEW_MARKER" in codes

def test_scan_reports_syntax_error_with_location_and_repair_hint(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "broken.py").write_text("def broken(:\n", encoding="utf-8")

    finding = next(
        f for f in scan_repository(tmp_path).findings
        if f.code == "PYTHON_SYNTAX" and f.path == "src/broken.py"
    )

    assert finding.severity == "ERROR"
    assert finding.line == 1
    assert finding.message
    assert "syntax error" in finding.suggestion.lower()


def test_capabilities_command_supports_json_output(tmp_path, capsys):
    assert main(["capabilities", str(tmp_path), "--json"]) == 0
    output = capsys.readouterr().out
    assert '"name": "core_reasoning"' in output
    assert '"status": "not_detected"' in output

def test_scan_rejects_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError): scan_repository(tmp_path / "missing")

def test_suggestions_are_deterministic():
    assert suggest_commands("pytest failing regression")[0][0] == "doctor tests"
    assert suggest_commands("niche issue")[0][0] == "doctor check"
    with pytest.raises(ValueError): suggest_commands(" ")

def test_scaffold_is_review_required_and_does_not_overwrite(tmp_path):
    target = scaffold_command("profile_memory", "Profile memory use", tmp_path)
    assert target == tmp_path / "src" / "doctor_commands" / "profile_memory.py"
    assert "NotImplementedError" in target.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError): scaffold_command("profile_memory", "Duplicate", tmp_path)

@pytest.mark.parametrize("name", ["../escape", "BadName", "x", "bad-name", "a" * 50])
def test_scaffold_rejects_unsafe_names(tmp_path, name):
    with pytest.raises(ValueError): scaffold_command(name, "description", tmp_path)


@pytest.mark.parametrize("timeout", [0, -1, 3601, True, 1.5, "30"])
def test_doctor_test_runner_rejects_invalid_timeout(tmp_path, timeout):
    from src.self_test import run_test_suite

    with pytest.raises(ValueError, match="timeout"):
        run_test_suite(tmp_path, timeout=timeout)


def test_scan_reports_syntax_error_only_once(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "broken.py").write_text("def broken(:\n", encoding="utf-8")

    findings = [
        finding for finding in scan_repository(tmp_path).findings
        if finding.code == "PYTHON_SYNTAX" and finding.path == "src/broken.py"
    ]
    assert len(findings) == 1
    assert findings[0].line == 1


def test_doctor_test_runner_timeout_preserves_readable_partial_output(tmp_path, monkeypatch):
    import subprocess
    from src.self_test import run_test_suite

    def timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs["timeout"],
            output=b"2 passed\n",
            stderr=b"warning: partial output\n",
        )

    monkeypatch.setattr(subprocess, "run", timeout_run)
    result = run_test_suite(tmp_path, timeout=5)

    assert result.returncode == 124
    assert result.stdout == "2 passed\n"
    assert "warning: partial output" in result.stderr
    assert "pytest timed out" in result.stderr
    assert "b'2 passed" not in result.stdout
