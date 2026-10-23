from src.repository_audit import RepositoryAuditor


def test_auditor_catches_syntax_errors_and_duplicate_pytest_workflows(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / ".github" / "workflows").mkdir(parents=True)

    for relative in RepositoryAuditor.REQUIRED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("" if not relative.endswith(".txt") else "pytest\n", encoding="utf-8")

    (tmp_path / "src" / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    (tmp_path / ".github" / "workflows" / "a.yml").write_text("run: python -m pytest -q\n", encoding="utf-8")
    (tmp_path / ".github" / "workflows" / "b.yml").write_text("run: python -m pytest -q\n", encoding="utf-8")

    report = RepositoryAuditor(tmp_path).audit()
    codes = {issue.code for issue in report.issues}
    assert "PYTHON_SYNTAX" in codes
    assert "DUPLICATE_PYTEST_WORKFLOWS" in codes
    assert not report.healthy


def test_clean_repository_shape_is_healthy(tmp_path):
    for relative in RepositoryAuditor.REQUIRED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pytest\n" if relative == "requirements.txt" else "# valid\n", encoding="utf-8")
    report = RepositoryAuditor(tmp_path).audit()
    assert report.healthy
    assert report.fingerprint


def test_auditor_flags_builtin_hash_for_persisted_logic(tmp_path):
    for relative in RepositoryAuditor.REQUIRED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pytest\n" if relative == "requirements.txt" else "# valid\n", encoding="utf-8")
    (tmp_path / "src" / "bad.py").write_text("value = hash('x')\n", encoding="utf-8")
    report = RepositoryAuditor(tmp_path).audit()
    issues = [issue for issue in report.issues if issue.code == "NONDETERMINISTIC_HASH"]
    assert issues
    assert issues[0].severity == "warning"



def test_audit_warnings_do_not_fail_health_status(tmp_path):
    for relative in RepositoryAuditor.REQUIRED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pytest\n" if relative == "requirements.txt" else "# valid\n", encoding="utf-8")
    (tmp_path / "src" / "warning.py").write_text("value = hash('x')\n", encoding="utf-8")

    report = RepositoryAuditor(tmp_path).audit()

    assert report.healthy
    assert any(issue.code == "NONDETERMINISTIC_HASH" and issue.severity == "warning" for issue in report.issues)


def test_audit_ignores_virtualenv_and_python_cache_files(tmp_path):
    for relative in RepositoryAuditor.REQUIRED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pytest\n" if relative == "requirements.txt" else "# valid\n", encoding="utf-8")
    before = RepositoryAuditor(tmp_path).audit().python_files

    ignored_files = [
        tmp_path / ".venv" / "lib" / "bad.py",
        tmp_path / "src" / "__pycache__" / "bad.py",
        tmp_path / "venv" / "bad.py",
    ]
    for path in ignored_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def broken(:\n", encoding="utf-8")

    report = RepositoryAuditor(tmp_path).audit()

    assert report.python_files == before
    assert not any(issue.path.endswith("bad.py") for issue in report.issues)


def test_audit_fingerprint_is_stable_for_the_same_repository(tmp_path):
    for relative in RepositoryAuditor.REQUIRED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pytest\n" if relative == "requirements.txt" else "# valid\n", encoding="utf-8")
    (tmp_path / "src" / "broken.py").write_text("def broken(:\n", encoding="utf-8")

    first = RepositoryAuditor(tmp_path).audit()
    second = RepositoryAuditor(tmp_path).audit()

    assert first.fingerprint == second.fingerprint
    assert first.issues == second.issues


def test_auditor_requires_existing_directory(tmp_path):
    import pytest

    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        RepositoryAuditor(missing)
