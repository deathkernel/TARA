from pathlib import Path
import subprocess

from src.self_developer import SelfDeveloper


def test_nested_glob_ignored_generated_test_is_validated_without_sweeping_artifacts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args, cwd=repo):
        subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    git("init")
    git("config", "user.email", "tara-tests@example.invalid")
    git("config", "user.name", "TARA test")

    (repo / ".gitignore").write_text(
        "tests/**/test_generated_*.py\noutputs/\n",
        encoding="utf-8",
    )
    (repo / "src").mkdir()
    (repo / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    git("add", ".gitignore", "src/module.py")
    git("commit", "-m", "test baseline with ignored generated-test glob")

    patch = (
        "diff --git a/tests/generated/test_generated_ignored.py "
        "b/tests/generated/test_generated_ignored.py\n"
        "new file mode 100644\n"
        "index 0000000..1111111\n"
        "--- /dev/null\n"
        "+++ b/tests/generated/test_generated_ignored.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def test_generated_file_is_executed():\n"
        "+    assert True\n"
    )

    class GeneratedTestProvider:
        def complete(self, *args, **kwargs):
            return patch

    def create_unrelated_ignored_artifact(args, **kwargs):
        result = subprocess.run(args, **kwargs)
        if args[:2] == ["git", "apply"] and "--check" not in args and result.returncode == 0:
            artifact = Path(kwargs["cwd"]) / "outputs" / "unrelated.txt"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("MUST_NOT_APPEAR_IN_CANDIDATE_DIFF", encoding="utf-8")
        return result

    engine = SelfDeveloper(
        repo,
        provider=GeneratedTestProvider(),
        state_root=tmp_path / "state",
        command_runner=create_unrelated_ignored_artifact,
    )
    report = engine.run_task("Validate a nested test matching a .gitignore glob", max_attempts=1)

    candidate = Path(report["candidate_worktree"])
    generated_test = "tests/generated/test_generated_ignored.py"

    assert report["status"] == "candidate_ready_for_review"
    assert generated_test in report["changed_files"]
    assert "outputs/unrelated.txt" not in report["changed_files"]
    assert "MUST_NOT_APPEAR_IN_CANDIDATE_DIFF" not in report["diff"]
    assert f"pytest -q {generated_test}" in report["attempts"][0]["test_output"]
    assert report["attempts"][0]["tests_passed"] is True
    assert (candidate / generated_test).is_file()
    assert (candidate / "outputs" / "unrelated.txt").is_file()
    assert subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", generated_test],
        cwd=candidate,
        check=False,
    ).returncode == 0
    assert not (repo / generated_test).exists()
    assert subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout == ""
