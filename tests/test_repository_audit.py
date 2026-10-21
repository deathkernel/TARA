from src.repository_audit import RepositoryAuditor


def _seed_required_files(root):
    for relative in RepositoryAuditor.REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "pytest\n" if relative == "requirements.txt" else "# valid\n",
            encoding="utf-8",
        )


def test_auditor_catches_syntax_errors_and_duplicate_pytest_workflows(tmp_path):
    _seed_required_files(tmp_path)
    (tmp_path / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    (tmp_path / ".github" / "workflows" / "a.yml").write_text(
        "run: python -m pytest -q\n", encoding="utf-8"
    )
    (tmp_path / ".github" / "workflows" / "b.yml").write_text(
        "run: python -m pytest -q\n", encoding="utf-8"
    )

    report = RepositoryAuditor(tmp_path).audit()
    codes = {issue.code for issue in report.issues}
    assert "PYTHON_SYNTAX" in codes
    assert "DUPLICATE_PYTEST_WORKFLOWS" in codes
    assert not report.healthy


def test_clean_repository_shape_is_healthy(tmp_path):
    _seed_required_files(tmp_path)
    report = RepositoryAuditor(tmp_path).audit()
    assert report.healthy
    assert report.fingerprint


def test_auditor_flags_builtin_hash_for_persisted_logic(tmp_path):
    _seed_required_files(tmp_path)
    (tmp_path / "src" / "bad.py").write_text("value = hash('x')\n", encoding="utf-8")
    report = RepositoryAuditor(tmp_path).audit()
    issues = [issue for issue in report.issues if issue.code == "NONDETERMINISTIC_HASH"]
    assert issues
    assert issues[0].severity == "warning"


def test_auditor_respects_root_gitignore_patterns(tmp_path):
    _seed_required_files(tmp_path)
    (tmp_path / ".gitignore").write_text(
        "generated/\nignored_*.py\n!ignored_keep.py\n", encoding="utf-8"
    )
    (tmp_path / "generated").mkdir()
    (tmp_path / "generated" / "generated_module.py").write_text(
        "this is invalid python (:\n", encoding="utf-8"
    )
    (tmp_path / "ignored_drop.py").write_text(
        "this is invalid python (:\n", encoding="utf-8"
    )
    (tmp_path / "ignored_keep.py").write_text("# should be included\n", encoding="utf-8")

    report = RepositoryAuditor(tmp_path).audit()
    syntax_paths = {
        issue.path for issue in report.issues if issue.code == "PYTHON_SYNTAX"
    }
    assert "generated/generated_module.py" not in syntax_paths
    assert "ignored_drop.py" not in syntax_paths
    assert "ignored_keep.py" not in syntax_paths


def test_auditor_respects_nested_gitignore_and_builtin_exclusions(tmp_path):
    _seed_required_files(tmp_path)
    nested = tmp_path / "src" / "generated"
    nested.mkdir()
    (tmp_path / "src" / ".gitignore").write_text("generated/\n", encoding="utf-8")
    (nested / "invalid.py").write_text("not python (:\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "invalid.py").write_text("not python (:\n", encoding="utf-8")

    report = RepositoryAuditor(tmp_path).audit()
    syntax_paths = {
        issue.path for issue in report.issues if issue.code == "PYTHON_SYNTAX"
    }
    assert "src/generated/invalid.py" not in syntax_paths
    assert ".venv/invalid.py" not in syntax_paths



def test_root_anchored_gitignore_patterns_do_not_match_nested_paths(tmp_path):
    _seed_required_files(tmp_path)
    (tmp_path / ".gitignore").write_text(
        "/ignored.py\n/src/only.py\n", encoding="utf-8"
    )
    (tmp_path / "ignored.py").write_text("invalid python (: \n", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "ignored.py").write_text(
        "invalid python (: \n", encoding="utf-8"
    )
    (tmp_path / "src" / "only.py").write_text(
        "invalid python (: \n", encoding="utf-8"
    )
    (tmp_path / "nested" / "src").mkdir()
    (tmp_path / "nested" / "src" / "only.py").write_text(
        "invalid python (: \n", encoding="utf-8"
    )

    report = RepositoryAuditor(tmp_path).audit()
    syntax_paths = {
        issue.path for issue in report.issues if issue.code == "PYTHON_SYNTAX"
    }
    assert "ignored.py" not in syntax_paths
    assert "src/only.py" not in syntax_paths
    assert "nested/ignored.py" in syntax_paths
    assert "nested/src/only.py" in syntax_paths
