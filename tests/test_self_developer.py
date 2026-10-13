import json
from pathlib import Path

import pytest

from src.self_developer import SelfDeveloper, SelfDeveloperError


@pytest.mark.parametrize("patch", [
    "",
    "```diff\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-a\n+b\n```",
    "--- a/../secret.txt\n+++ b/../secret.txt\n@@ -1 +1 @@\n-a\n+b\n",
    "--- a/checkpoints/model.pt\n+++ b/checkpoints/model.pt\n@@ -1 +1 @@\n-a\n+b\n",
    "--- a/.github/workflows/tests.yml\n+++ b/.github/workflows/tests.yml\n@@ -1 +1 @@\n-a\n+b\n",
    "--- a/requirements.txt\n+++ b/requirements.txt\n@@ -1 +1 @@\n-a\n+b\n",
    "diff --git a/../../outside.txt b/src/example.py\n--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-a\n+b\n",
    "diff --git a/src/one.py b/src/one.py\n--- a/src/two.py\n+++ b/src/one.py\n@@ -1 +1 @@\n-a\n+b\n",
    "--- a/.git\n+++ b/.git\n@@ -1 +1 @@\n-a\n+b\n",
    "--- a/src/link\n+++ b/src/link\nnew file mode 120000\n@@ -0,0 +1 @@\n+target\n",
    "--- a/src/blob.bin\n+++ b/src/blob.bin\nGIT binary patch\n@@ -0,0 +1 @@\n+binary\n",
])
def test_patch_guard_rejects_unsafe_or_invalid_patch(patch):
    with pytest.raises(SelfDeveloperError):
        SelfDeveloper._validate_patch(patch)


def test_patch_guard_accepts_small_repository_relative_diff():
    patch = "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-old\n+new\n"
    SelfDeveloper._validate_patch(patch)


def test_provider_is_required_for_planning(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    engine = SelfDeveloper(repo, state_root=tmp_path / "state")
    with pytest.raises(SelfDeveloperError, match="LLM provider"):
        engine.plan("Add a test")


def test_report_is_written_and_can_be_read(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    engine = SelfDeveloper(repo, state_root=tmp_path / "state")
    written = engine._write_report({"run_id": "test-run", "status": "plan_only", "task": "test"})
    assert Path(written["report_path"]).is_file()
    latest = engine.latest_report()
    assert latest["task"] == "test"
    assert latest["report_path"] == written["report_path"]
    persisted = json.loads(Path(written["report_path"]).read_text(encoding="utf-8"))
    assert persisted["status"] == "plan_only"
    assert persisted["report_path"] == written["report_path"]



def test_command_runner_disables_shell_and_uses_timeout(tmp_path):
    from types import SimpleNamespace

    from src.self_developer import _run

    observed = {}

    def fake_runner(args, **kwargs):
        observed["args"] = args
        observed.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    result = _run(["git", "status"], cwd=tmp_path, timeout=17, runner=fake_runner)
    assert result.stdout == "ok"
    assert observed["shell"] is False
    assert observed["timeout"] == 17
    assert observed["args"] == ["git", "status"]


def test_command_runner_converts_timeout_to_controlled_error(tmp_path):
    import subprocess

    from src.self_developer import _run

    def timeout_runner(args, **kwargs):
        raise subprocess.TimeoutExpired(args=args, timeout=kwargs["timeout"])

    with pytest.raises(SelfDeveloperError, match="timed out after 9s"):
        _run(["pytest"], cwd=tmp_path, timeout=9, runner=timeout_runner)


def test_command_runner_bounds_failure_output(tmp_path):
    from types import SimpleNamespace

    from src.self_developer import _OUTPUT_LIMIT, _run

    def failed_runner(args, **kwargs):
        return SimpleNamespace(returncode=2, stdout="", stderr="x" * (_OUTPUT_LIMIT + 100))

    with pytest.raises(SelfDeveloperError) as exc:
        _run(["pytest"], cwd=tmp_path, runner=failed_runner)
    assert len(str(exc.value)) < _OUTPUT_LIMIT + 200

def test_task_validation_rejects_empty_and_oversized_tasks():
    with pytest.raises(ValueError):
        SelfDeveloper._validate_task(" ")
    with pytest.raises(ValueError):
        SelfDeveloper._validate_task("x" * 2001)


def test_command_runner_converts_os_error_to_controlled_error(tmp_path):
    from src.self_developer import _run

    def missing_command(args, **kwargs):
        raise FileNotFoundError("git executable missing")

    with pytest.raises(SelfDeveloperError, match="Could not start git"):
        _run(["git", "status"], cwd=tmp_path, runner=missing_command)


def test_worktree_creation_failure_is_reported_without_provider_call(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(
            ["git", *args], cwd=repo, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    git("init")
    git("config", "user.email", "tara-tests@example.invalid")
    git("config", "user.name", "TARA test")
    (repo / "README.md").write_text("baseline\\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-m", "test baseline")

    class NeverCalledProvider:
        def complete(self, *args, **kwargs):
            raise AssertionError("provider must not run when worktree creation fails")

    def fail_worktree_add(args, **kwargs):
        if args[:3] == ["git", "worktree", "add"]:
            raise OSError("simulated worktree creation failure")
        return subprocess.run(args, **kwargs)

    engine = SelfDeveloper(
        repo, provider=NeverCalledProvider(), state_root=tmp_path / "state",
        command_runner=fail_worktree_add,
    )
    report = engine.run_task("Add a focused regression test")

    assert report["status"] == "failed"
    assert "simulated worktree creation failure" in report["error"]
    assert report["attempts"] == []
    assert Path(report["report_path"]).is_file()
    assert not Path(report["candidate_worktree"]).exists()
    assert subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout == ""


def test_invalid_patch_retries_are_bounded_and_leave_base_unchanged(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(
            ["git", *args], cwd=repo, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    git("init")
    git("config", "user.email", "tara-tests@example.invalid")
    git("config", "user.name", "TARA test")
    (repo / "baseline.py").write_text("VALUE = 1\\n", encoding="utf-8")
    git("add", "baseline.py")
    git("commit", "-m", "test baseline")

    class EmptyPatchProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, *args, **kwargs):
            self.calls += 1
            return ""

    provider = EmptyPatchProvider()
    engine = SelfDeveloper(repo, provider=provider, state_root=tmp_path / "state")
    report = engine.run_task("Make a small safe change", max_attempts=2)

    assert provider.calls == 2
    assert report["status"] == "failed"
    assert len(report["attempts"]) == 2
    assert all(not attempt["tests_passed"] for attempt in report["attempts"])
    assert "All bounded attempts failed" in report["error"]
    assert (repo / "baseline.py").read_text(encoding="utf-8") == "VALUE = 1\\n"
    assert not Path(report["candidate_worktree"]).exists()
    assert subprocess.run(
        ["git", "branch", "--list", report["candidate_branch"]], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip() == ""
    assert subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout == ""


def test_provider_exception_cleans_failed_candidate_and_keeps_report(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(
            ["git", *args], cwd=repo, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    git("init")
    git("config", "user.email", "tara-tests@example.invalid")
    git("config", "user.name", "TARA test")
    (repo / "baseline.py").write_text("VALUE = 1\\n", encoding="utf-8")
    git("add", "baseline.py")
    git("commit", "-m", "test baseline")

    class ExplodingProvider:
        def complete(self, *args, **kwargs):
            raise RuntimeError("simulated provider outage")

    engine = SelfDeveloper(repo, provider=ExplodingProvider(), state_root=tmp_path / "state")
    report = engine.run_task("Add a focused test", max_attempts=1)

    assert report["status"] == "failed"
    assert "simulated provider outage" in report["error"]
    assert Path(report["report_path"]).is_file()
    assert not Path(report["candidate_worktree"]).exists()
    assert subprocess.run(
        ["git", "branch", "--list", report["candidate_branch"]], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip() == ""
    assert subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout == ""


def test_successful_candidate_is_retained_for_manual_review(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(
            ["git", *args], cwd=repo, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    git("init")
    git("config", "user.email", "tara-tests@example.invalid")
    git("config", "user.name", "TARA test")
    (repo / "src").mkdir()
    (repo / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    git("add", "src/module.py")
    git("commit", "-m", "test baseline")

    class ValidPatchProvider:
        def complete(self, *args, **kwargs):
            return "--- a/src/module.py\n+++ b/src/module.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2"

    engine = SelfDeveloper(repo, provider=ValidPatchProvider(), state_root=tmp_path / "state")
    report = engine.run_task("Change module value", max_attempts=1)

    candidate = Path(report["candidate_worktree"])
    assert report["status"] == "candidate_ready_for_review"
    assert candidate.is_dir()
    assert (candidate / "src" / "module.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (repo / "src" / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert subprocess.run(
        ["git", "branch", "--list", report["candidate_branch"]], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip() == report["candidate_branch"]
    assert subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout == ""
