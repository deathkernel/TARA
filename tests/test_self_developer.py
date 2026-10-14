import json
from pathlib import Path

import pytest

from src.self_developer import SelfDeveloper, SelfDeveloperError, _validation_commands


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
    "--- a/src/.git/config\n+++ b/src/.git/config\n@@ -1 +1 @@\n-a\n+b\n",
    "--- a/requirements.txt\t2026-10-13 10:00:00 +0000\n+++ b/requirements.txt\t2026-10-13 10:00:00 +0000\n@@ -1 +1 @@\n-old\n+new\n",
    "--- a/src/link\n+++ b/src/link\nnew file mode 120000\n@@ -0,0 +1 @@\n+target\n",
    "--- a/src/blob.bin\n+++ b/src/blob.bin\nGIT binary patch\n@@ -0,0 +1 @@\n+binary\n",
])
def test_patch_guard_rejects_unsafe_or_invalid_patch(patch):
    with pytest.raises(SelfDeveloperError):
        SelfDeveloper._validate_patch(patch)


def test_patch_guard_accepts_small_repository_relative_diff():
    patch = "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-old\n+new\n"
    SelfDeveloper._validate_patch(patch)


@pytest.mark.parametrize("path", [
    "src/CON",
    "src/nul.txt",
    "src/AUX.log",
    "src/COM1.py",
    "src/LPT9.data",
    "src/COM¹.txt",
    "src/file:stream.py",
    "src/bad<name.py",
    "src/bad>name.py",
    "src/bad|name.py",
    "src/bad?name.py",
    "src/bad*name.py",
    "src/bad\u0001name.py",
    "src/trailing.",
    "src/trailing ",
])
def test_patch_guard_rejects_windows_special_paths(path):
    patch = f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-old\n+new\n"
    with pytest.raises(SelfDeveloperError):
        SelfDeveloper._validate_patch(patch)


@pytest.mark.parametrize("path", [
    "src/console.py",
    "src/auxiliary.py",
    "src/com10.py",
    "src/lpt0.py",
])
def test_patch_guard_allows_non_reserved_windows_lookalikes(path):
    patch = f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-old\n+new\n"
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

def test_validation_commands_run_full_suite_for_source_changes():
    commands = _validation_commands(["src/brain.py"])

    assert commands[0][1:] == ["-m", "compileall", "-q", "src"]
    assert commands[1][1:] == ["-m", "pytest", "-q"]


def test_validation_commands_target_changed_tests_when_only_tests_change():
    commands = _validation_commands(["tests/test_memory.py"])

    assert commands[0][1:] == ["-m", "compileall", "-q", "src"]
    assert commands[1][1:] == ["-m", "pytest", "-q", "tests/test_memory.py"]


def test_validation_commands_run_full_suite_for_mixed_source_and_test_changes():
    commands = _validation_commands(["src/memory.py", "tests/test_memory.py"])

    assert commands[1][1:] == ["-m", "pytest", "-q"]


def test_validation_commands_compile_source_when_no_python_source_or_tests_change():
    commands = _validation_commands(["README.md", "docs/guide.md"])

    assert len(commands) == 1
    assert commands[0][1:] == ["-m", "compileall", "-q", "src"]

def test_validation_commands_compile_changed_root_python_scripts():
    commands = _validation_commands(["train_tiny_lm.py"])

    assert commands[0][1:] == ["-m", "compileall", "-q", "src"]
    assert commands[1][1:] == ["-m", "py_compile", "train_tiny_lm.py"]


def test_validation_commands_compile_root_python_scripts_alongside_changed_tests():
    commands = _validation_commands(["tests/test_memory.py", "prepare_tara_dataset.py"])

    assert commands[1][1:] == ["-m", "pytest", "-q", "tests/test_memory.py"]
    assert commands[2][1:] == ["-m", "py_compile", "prepare_tara_dataset.py"]


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
    (repo / "README.md").write_text("baseline\n", encoding="utf-8")
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
    (repo / "baseline.py").write_text("VALUE = 1\n", encoding="utf-8")
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
    assert (repo / "baseline.py").read_text(encoding="utf-8") == "VALUE = 1\n"
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
    (repo / "baseline.py").write_text("VALUE = 1\n", encoding="utf-8")
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


def test_failed_test_attempt_rolls_back_before_retry(tmp_path):
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
    (repo / "tests").mkdir()
    (repo / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    git("add", "src/module.py")
    git("commit", "-m", "test baseline")

    class RetryProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return "--- /dev/null\n+++ b/tests/test_broken.py\n@@ -0,0 +1,2 @@\n+def test_intentionally_fails():\n+    assert False\n"
            return "--- a/src/module.py\n+++ b/src/module.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2"

    provider = RetryProvider()
    engine = SelfDeveloper(repo, provider=provider, state_root=tmp_path / "state")
    report = engine.run_task("Exercise rollback and retry", max_attempts=2)

    candidate = Path(report["candidate_worktree"])
    assert provider.calls == 2
    assert report["status"] == "candidate_ready_for_review"
    assert len(report["attempts"]) == 2
    assert report["attempts"][0]["tests_passed"] is False
    assert "Command failed" in report["attempts"][0]["result"]
    assert report["attempts"][1]["tests_passed"] is True
    assert not (candidate / "tests" / "test_broken.py").exists()
    assert (candidate / "src" / "module.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (repo / "src" / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout == ""
def test_apply_check_failure_rolls_back_and_retries(tmp_path):
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

    class ApplyRetryProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return "--- a/src/module.py\n+++ b/src/module.py\n@@ -1 +1 @@\n-VALUE = 999\n+VALUE = 2"
            return "--- a/src/module.py\n+++ b/src/module.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2"

    provider = ApplyRetryProvider()
    engine = SelfDeveloper(repo, provider=provider, state_root=tmp_path / "state")
    report = engine.run_task("Retry a patch that does not apply", max_attempts=2)

    candidate = Path(report["candidate_worktree"])
    assert provider.calls == 2
    assert report["status"] == "candidate_ready_for_review"
    assert len(report["attempts"]) == 2
    assert report["attempts"][0]["tests_passed"] is False
    assert "Command failed" in report["attempts"][0]["result"]
    assert report["attempts"][1]["tests_passed"] is True
    assert (candidate / "src" / "module.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (repo / "src" / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout == ""


def test_cleanup_warnings_are_reported_and_failure_report_retained(tmp_path):
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
    (repo / "baseline.py").write_text("VALUE = 1\n", encoding="utf-8")
    git("add", "baseline.py")
    git("commit", "-m", "test baseline")

    class EmptyPatchProvider:
        def complete(self, *args, **kwargs):
            return ""

    def cleanup_failure_runner(args, **kwargs):
        if args[:3] == ["git", "worktree", "remove"]:
            raise OSError("simulated worktree cleanup failure")
        if args[:3] == ["git", "branch", "-D"]:
            raise OSError("simulated branch cleanup failure")
        return subprocess.run(args, **kwargs)

    engine = SelfDeveloper(
        repo, provider=EmptyPatchProvider(), state_root=tmp_path / "state",
        command_runner=cleanup_failure_runner,
    )
    report = engine.run_task("Keep cleanup failures visible", max_attempts=1)

    assert report["status"] == "failed"
    assert len(report["cleanup_warnings"]) == 2
    assert any("worktree cleanup failure" in item for item in report["cleanup_warnings"])
    assert any("branch cleanup failure" in item for item in report["cleanup_warnings"])
    report_path = Path(report["report_path"])
    assert report_path.is_file()
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["cleanup_warnings"] == report["cleanup_warnings"]
    assert persisted["status"] == "failed"


def test_standard_git_diff_can_add_regular_test_file_in_candidate_only(tmp_path):
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

    patch = (
        "diff --git a/tests/test_added.py b/tests/test_added.py\n"
        "new file mode 100644\n"
        "index 0000000..1111111\n"
        "--- /dev/null\n"
        "+++ b/tests/test_added.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def test_added_file_is_runnable():\n"
        "+    assert True\n"
    )

    class NewFileProvider:
        def complete(self, *args, **kwargs):
            return patch

    engine = SelfDeveloper(repo, provider=NewFileProvider(), state_root=tmp_path / "state")
    report = engine.run_task("Add a small focused regression test", max_attempts=1)

    candidate = Path(report["candidate_worktree"])
    assert report["status"] == "candidate_ready_for_review"
    assert report["attempts"][0]["tests_passed"] is True
    assert (candidate / "tests" / "test_added.py").is_file()
    assert not (repo / "tests" / "test_added.py").exists()
    assert subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout == ""


def test_context_uses_candidate_snapshot_not_dirty_original_checkout(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args, cwd=repo):
        return subprocess.run(
            ["git", *args], cwd=cwd, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    git("init")
    git("config", "user.email", "tara-tests@example.invalid")
    git("config", "user.name", "TARA test")
    (repo / "src").mkdir()
    source = repo / "src" / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    git("add", "src/module.py")
    git("commit", "-m", "test baseline")

    # Simulate a user edit in the original checkout after the candidate base HEAD.
    source.write_text("VALUE = 99\n", encoding="utf-8")
    candidate = tmp_path / "candidate"
    git("worktree", "add", "--detach", str(candidate), "HEAD")

    engine = SelfDeveloper(repo, state_root=tmp_path / "state")
    context = engine._tracked_context(candidate)

    assert "VALUE = 1" in context
    assert "VALUE = 99" not in context
    assert source.read_text(encoding="utf-8") == "VALUE = 99\n"
    assert (candidate / "src" / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    git("worktree", "remove", "--force", str(candidate))


def test_default_context_uses_committed_head_and_excludes_staged_files(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    git("init")
    git("config", "user.email", "tara-tests@example.invalid")
    git("config", "user.name", "TARA test")
    (repo / "src").mkdir()
    source = repo / "src" / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    git("add", "src/module.py")
    git("commit", "-m", "test baseline")

    source.write_text("SECRET_UNCOMMITTED_VALUE = 'do-not-send'\n", encoding="utf-8")
    (repo / "staged_private.py").write_text("PRIVATE_STAGED_CONTENT = True\n", encoding="utf-8")
    git("add", "staged_private.py")

    engine = SelfDeveloper(repo, state_root=tmp_path / "state")
    context = engine._tracked_context()

    assert "VALUE = 1" in context
    assert "SECRET_UNCOMMITTED_VALUE" not in context
    assert "staged_private.py" not in context
    assert "PRIVATE_STAGED_CONTENT" not in context
    assert source.read_text(encoding="utf-8") == "SECRET_UNCOMMITTED_VALUE = 'do-not-send'\n"


def test_candidate_context_does_not_follow_symlinks_outside_worktree(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args, cwd=repo):
        return subprocess.run(
            ["git", *args], cwd=cwd, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    git("init")
    git("config", "user.email", "tara-tests@example.invalid")
    git("config", "user.name", "TARA test")
    (repo / "src").mkdir()
    (repo / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    outside_secret = tmp_path / "private-outside-worktree.py"
    outside_secret.write_text("PRIVATE_EXTERNAL_SECRET = 'must-not-enter-context'\n", encoding="utf-8")
    (repo / "src" / "external.py").symlink_to(outside_secret)
    git("add", "src/module.py", "src/external.py")
    git("commit", "-m", "test symlink context boundary")

    candidate = tmp_path / "candidate"
    git("worktree", "add", "--detach", str(candidate), "HEAD")
    engine = SelfDeveloper(repo, state_root=tmp_path / "state")
    context = engine._tracked_context(candidate)

    assert "VALUE = 1" in context
    assert "PRIVATE_EXTERNAL_SECRET" not in context
    assert "must-not-enter-context" not in context
    git("worktree", "remove", "--force", str(candidate))

def test_ignored_generated_test_is_discovered_without_sweeping_other_ignored_files(tmp_path):
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

    patch = (
        "diff --git a/.gitignore b/.gitignore\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/.gitignore\n"
        "@@ -0,0 +1,2 @@\n"
        "+tests/test_generated.py\n"
        "+outputs/\n"
        "diff --git a/tests/test_generated.py b/tests/test_generated.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_generated.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def test_generated_file_is_validated():\n"
        "+    assert True\n"
    )

    class IgnoredTestProvider:
        def complete(self, *args, **kwargs):
            return patch

    def add_unrelated_ignored_artifact(args, **kwargs):
        result = subprocess.run(args, **kwargs)
        # Simulate a tool creating an unrelated ignored artifact after the
        # patch is applied but before the candidate's change list is collected.
        if args[:2] == ["git", "apply"] and "--check" not in args and result.returncode == 0:
            artifact = Path(kwargs["cwd"]) / "outputs" / "unrelated.txt"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("MUST_NOT_APPEAR_IN_CANDIDATE_DIFF", encoding="utf-8")
        return result

    engine = SelfDeveloper(
        repo, provider=IgnoredTestProvider(), state_root=tmp_path / "state",
        command_runner=add_unrelated_ignored_artifact,
    )
    report = engine.run_task("Add a test that is accidentally ignored", max_attempts=1)

    candidate = Path(report["candidate_worktree"])
    assert report["status"] == "candidate_ready_for_review"
    assert "tests/test_generated.py" in report["changed_files"]
    assert "outputs/unrelated.txt" not in report["changed_files"]
    assert "MUST_NOT_APPEAR_IN_CANDIDATE_DIFF" not in report["diff"]
    assert report["attempts"][0]["tests_passed"] is True
    assert (candidate / "tests" / "test_generated.py").is_file()
    assert (candidate / "outputs" / "unrelated.txt").is_file()
    assert not (repo / "tests" / "test_generated.py").exists()
def test_reset_failure_stops_retries_and_removes_candidate(tmp_path):
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
    source = repo / "src" / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    git("add", "src/module.py")
    git("commit", "-m", "test baseline")

    class FailingTestProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, *args, **kwargs):
            self.calls += 1
            return (
                "--- /dev/null\n"
                "+++ b/tests/test_failure.py\n"
                "@@ -0,0 +1,2 @@\n"
                "+def test_intentionally_fails():\n"
                "+    assert False\n"
            )

    def reset_failure_runner(args, **kwargs):
        if args[:3] == ["git", "reset", "--hard"]:
            raise OSError("simulated candidate reset failure")
        return subprocess.run(args, **kwargs)

    provider = FailingTestProvider()
    engine = SelfDeveloper(
        repo, provider=provider, state_root=tmp_path / "state",
        command_runner=reset_failure_runner,
    )
    report = engine.run_task("Stop safely when candidate rollback fails", max_attempts=3)

    assert provider.calls == 1
    assert report["status"] == "failed"
    assert len(report["attempts"]) == 1
    assert "Candidate reset failed" in report["error"]
    assert "simulated candidate reset failure" in report["attempts"][0]["rollback_error"]
    assert not Path(report["candidate_worktree"]).exists()
    assert source.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout == ""


def test_report_write_preserves_preexisting_temp_neighbor(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    engine = SelfDeveloper(repo, state_root=tmp_path / "state")
    engine.runs_root.mkdir(parents=True)
    neighbor = engine.runs_root / "report-safety.json.tmp"
    neighbor.write_text("user-owned temp neighbor\n", encoding="utf-8")

    report = engine._write_report({"run_id": "report-safety", "status": "ok"})

    assert json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))["status"] == "ok"
    assert neighbor.read_text(encoding="utf-8") == "user-owned temp neighbor\n"
    assert list(engine.runs_root.glob(".report-safety.json.*.tmp")) == []


def test_report_write_failure_preserves_existing_report_and_temp_neighbor(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    engine = SelfDeveloper(repo, state_root=tmp_path / "state")
    engine.runs_root.mkdir(parents=True)
    destination = engine.runs_root / "report-safety.json"
    destination.write_text('{"status": "known-good"}\n', encoding="utf-8")
    neighbor = engine.runs_root / "report-safety.json.tmp"
    neighbor.write_text("user-owned temp neighbor\n", encoding="utf-8")

    with pytest.raises(TypeError):
        engine._write_report({
            "run_id": "report-safety",
            "status": "replacement",
            "not_json_serializable": object(),
        })

    assert destination.read_text(encoding="utf-8") == '{"status": "known-good"}\n'
    assert neighbor.read_text(encoding="utf-8") == "user-owned temp neighbor\n"
    assert list(engine.runs_root.glob(".report-safety.json.*.tmp")) == []
