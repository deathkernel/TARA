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
    assert engine.latest_report()["task"] == "test"
    assert json.loads(Path(written["report_path"]).read_text(encoding="utf-8"))["status"] == "plan_only"


def test_task_validation_rejects_empty_and_oversized_tasks():
    with pytest.raises(ValueError):
        SelfDeveloper._validate_task(" ")
    with pytest.raises(ValueError):
        SelfDeveloper._validate_task("x" * 2001)
