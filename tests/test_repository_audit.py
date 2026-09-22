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
