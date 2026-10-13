"""Controlled, repository-scoped self-development workflow for TARA.

The engine generates candidate diffs with an explicitly configured LLM, applies
them only in a separate Git worktree, runs a fixed test command, and writes an
audit report. It never merges, pushes, installs dependencies, or edits the
original working tree automatically.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROTECTED_PATHS = (
    "checkpoints/",
    "outputs/",
    "data/conversation_v1.jsonl",
    "data/conversation_v1.jsonl.manifest.json",
    "test_voice.py",
)
MAX_PATCH_CHARS = 100_000
MAX_CONTEXT_CHARS = 50_000
COMMAND_TIMEOUT_SECONDS = 300
_OUTPUT_LIMIT = 12_000


class SelfDeveloperError(RuntimeError):
    """A controlled self-development operation could not continue."""


def _default_state_root() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "TARA" / "self-developer"


def _run(args: list[str], *, cwd: Path, timeout: int = COMMAND_TIMEOUT_SECONDS,
         runner: Callable[..., Any] = subprocess.run) -> subprocess.CompletedProcess:
    """Run an explicit argument vector; never invoke a shell."""
    try:
        result = runner(
            args, cwd=str(cwd), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SelfDeveloperError(f"Command timed out after {timeout}s: {args[0]}") from exc
    except OSError as exc:
        raise SelfDeveloperError(f"Could not start {args[0]}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-_OUTPUT_LIMIT:]
        raise SelfDeveloperError(f"Command failed ({result.returncode}): {args[0]}: {detail}")
    return result


class SelfDeveloper:
    """Prepare, test, and report candidate changes without integrating them."""

    def __init__(self, repository: str | Path, *, provider=None,
                 state_root: str | Path | None = None,
                 command_runner: Callable[..., Any] = subprocess.run) -> None:
        self.repository = Path(repository).resolve()
        if not (self.repository / ".git").exists():
            if not (self.repository / ".git").is_file():
                raise FileNotFoundError(f"Not a Git repository: {self.repository}")
        self.provider = provider
        self.state_root = Path(state_root).resolve() if state_root else _default_state_root()
        self.runs_root = self.state_root / "runs"
        self.worktrees_root = self.state_root / "worktrees"
        self._command_runner = command_runner

    def _git(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
        return _run(["git", *args], cwd=cwd or self.repository, runner=self._command_runner)

    def status(self) -> dict[str, Any]:
        branch = self._git("branch", "--show-current").stdout.strip()
        head = self._git("rev-parse", "--short", "HEAD").stdout.strip()
        status = self._git("status", "--short").stdout
        return {
            "repository": str(self.repository),
            "branch": branch,
            "head": head,
            "working_tree_status": status.splitlines(),
            "candidate_worktrees_root": str(self.worktrees_root),
            "reports_root": str(self.runs_root),
            "integration": "manual approval required",
        }

    def _require_provider(self):
        if self.provider is None:
            raise SelfDeveloperError(
                'LLM provider is not configured. In CMD run: set TARA_LLM_BACKEND=ollama'
            )
        return self.provider

    def _tracked_context(self) -> str:
        paths = self._git("ls-files").stdout.splitlines()
        chunks = ["Tracked repository paths:\n" + "\n".join(paths[:500])]
        budget = MAX_CONTEXT_CHARS - len(chunks[0])
        preferred = [
            "src/brain.py", "src/agent.py", "src/memory.py",
            "src/safety.py", "src/tools/controller.py",
            "src/llm_provider_factory.py", "src/ollama_provider.py",
        ]
        selected = [p for p in preferred if p in paths]
        selected.extend(p for p in paths if p.startswith(("src/", "tests/"))
                        and p.endswith(".py") and p not in selected)
        for relative in selected:
            if budget <= 0 or len(chunks) >= 24:
                break
            try:
                content = (self.repository / relative).read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            if len(content) > 8_000:
                content = content[:8_000] + "\n[truncated]"
            chunk = f"\n--- FILE: {relative} ---\n{content}\n"
            if len(chunk) > budget:
                continue
            chunks.append(chunk)
            budget -= len(chunk)
        return "\n".join(chunks)

    def plan(self, task: str) -> dict[str, Any]:
        task = self._validate_task(task)
        provider = self._require_provider()
        prompt = (
            "Create a conservative implementation plan for this task in the TARA "
            "repository. Do not claim files were changed or tests were run. "
            "List relevant files, steps, tests, risks, and rollback considerations. "
            "Respect protected user artifacts and never propose merge/push. Task:\n"
            + task + "\n\nRepository context:\n" + self._tracked_context()
        )
        plan_text = provider.complete(
            prompt,
            system="You are TARA's cautious software engineering planner. Be specific and honest.",
            max_completion_tokens=900,
            temperature=0.2,
        ).strip()
        return self._write_report({
            "task": task,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "plan": plan_text,
            "status": "plan_only",
            "integration": "requires explicit user approval",
        })

    def _cleanup_failed_candidate(self, worktree: Path, branch: str) -> list[str]:
        """Remove only this run's failed candidate worktree/branch; retain its report."""
        warnings: list[str] = []
        if not branch.startswith("tara-selfdev-"):
            return ["refused cleanup for a branch outside the self-developer namespace"]
        if worktree.exists():
            try:
                self._git("worktree", "remove", "--force", str(worktree))
            except SelfDeveloperError as exc:
                warnings.append(f"could not remove failed worktree: {exc}")
        try:
            self._git("branch", "-D", branch)
        except SelfDeveloperError as exc:
            # Worktree creation can fail before Git creates the branch.
            if "not found" not in str(exc).lower() and "not exist" not in str(exc).lower():
                warnings.append(f"could not remove failed candidate branch: {exc}")
        return warnings

    def run_task(self, task: str, *, max_attempts: int = 3) -> dict[str, Any]:
        task = self._validate_task(task)
        if not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")
        provider = self._require_provider()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        branch = "tara-selfdev-" + run_id.lower()
        worktree = self.worktrees_root / run_id
        self.worktrees_root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        base_head = self._git("rev-parse", "HEAD").stdout.strip()
        record: dict[str, Any] = {
            "run_id": run_id,
            "task": task,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_branch": self._git("branch", "--show-current").stdout.strip(),
            "base_head": base_head,
            "candidate_branch": branch,
            "candidate_worktree": str(worktree),
            "attempts": [],
            "status": "started",
            "integration": "not performed; approval required",
        }
        try:
            self._git("worktree", "add", "-b", branch, str(worktree), base_head)
            context = self._tracked_context()
            for attempt in range(1, max_attempts + 1):
                prompt = (
                    "Implement the requested task in the TARA repository. Return ONLY a "
                    "unified git diff with paths relative to repository root; no Markdown "
                    "fences or prose. Do not change protected artifacts, dependency files, "
                    "workflow files, secrets, or files outside the repository. Keep the patch "
                    "small and add/update focused tests. Never claim tests were run. If unable "
                    "to solve it, return an empty response.\n\nTASK:\n" + task
                    + "\n\nREPOSITORY CONTEXT:\n" + context
                )
                if record["attempts"]:
                    previous = record["attempts"][-1]
                    prompt += (
                        "\n\nPrevious attempt failed validation/testing. Fix it. "
                        "Last result:\n" + str(previous.get("result", ""))[-6000:]
                    )
                patch = provider.complete(
                    prompt,
                    system="You are a careful patch generator. Output only a valid unified diff.",
                    max_completion_tokens=3500,
                    temperature=0.1,
                ).strip()
                attempt_record: dict[str, Any] = {"attempt": attempt, "patch_chars": len(patch)}
                patch_file = None
                applied = False
                try:
                    self._validate_patch(patch)
                    patch_file = self.runs_root / f"{run_id}.attempt-{attempt}.patch"
                    patch_file.write_text(patch + "\n", encoding="utf-8")
                    self._git("apply", "--check", str(patch_file), cwd=worktree)
                    self._git("apply", str(patch_file), cwd=worktree)
                    applied = True
                    # Include newly-created files in diff/test discovery without staging content.
                    self._git("add", "-N", "--", ".", cwd=worktree)
                    self._git("diff", "--check", cwd=worktree)
                    changed = self._git("diff", "--name-only", cwd=worktree).stdout.splitlines()
                    tests = [p for p in changed if p.startswith("tests/") and p.endswith(".py")]
                    if tests:
                        test_result = _run(
                            [sys.executable, "-m", "pytest", "-q", *tests],
                            cwd=worktree, timeout=COMMAND_TIMEOUT_SECONDS,
                            runner=self._command_runner,
                        )
                    else:
                        test_result = _run(
                            [sys.executable, "-m", "compileall", "-q", "src"],
                            cwd=worktree, timeout=COMMAND_TIMEOUT_SECONDS,
                            runner=self._command_runner,
                        )
                    attempt_record.update({
                        "changed_files": changed,
                        "tests_passed": True,
                        "test_output": (test_result.stdout or "")[-_OUTPUT_LIMIT:],
                        "result": "validation and selected checks passed",
                    })
                    if patch_file is not None:
                        patch_file.unlink(missing_ok=True)
                    record["attempts"].append(attempt_record)
                    record["changed_files"] = changed
                    record["status"] = "candidate_ready_for_review"
                    record["diff"] = self._git("diff", "--", *changed, cwd=worktree).stdout[:MAX_PATCH_CHARS]
                    return self._write_report(record)
                except (SelfDeveloperError, OSError, ValueError) as exc:
                    attempt_record.update({"tests_passed": False, "result": str(exc)})
                    record["attempts"].append(attempt_record)
                    # Candidate-only cleanup; never reset or clean the original working tree.
                    if applied and patch_file is not None and patch_file.exists():
                        try:
                            self._git("apply", "--reverse", str(patch_file), cwd=worktree)
                        except Exception:
                            pass
                    try:
                        self._git("reset", "--hard", base_head, cwd=worktree)
                    except SelfDeveloperError:
                        pass
                    if patch_file is not None:
                        patch_file.unlink(missing_ok=True)
            record["status"] = "failed"
            record["error"] = "All bounded attempts failed validation or checks."
            cleanup_warnings = self._cleanup_failed_candidate(worktree, branch)
            if cleanup_warnings:
                record["cleanup_warnings"] = cleanup_warnings
            return self._write_report(record)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            cleanup_warnings = self._cleanup_failed_candidate(worktree, branch)
            if cleanup_warnings:
                record["cleanup_warnings"] = cleanup_warnings
            return self._write_report(record)

    @staticmethod
    def _validate_task(task: str) -> str:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must not be empty")
        task = task.strip()
        if len(task) > 2_000:
            raise ValueError("task must be 2000 characters or fewer")
        return task

    @staticmethod
    def _validate_patch(patch: str) -> None:
        if not patch or not patch.strip():
            raise SelfDeveloperError("LLM returned an empty patch")
        if len(patch) > MAX_PATCH_CHARS:
            raise SelfDeveloperError(f"Patch exceeds {MAX_PATCH_CHARS} characters")
        if "```" in patch:
            raise SelfDeveloperError("Patch must not contain Markdown fences")
        paths = []
        lines = patch.splitlines()
        for line in lines:
            if line.startswith("diff --git "):
                match = re.match(r"^diff --git a/(.*?) b/(.*?)$", line)
                if not match:
                    raise SelfDeveloperError("Malformed or unsupported git diff header")
                paths.extend(item.replace("\\", "/") for item in match.groups())
            elif line.startswith("--- a/"):
                paths.append(line[6:].split("\t", 1)[0].replace("\\", "/"))
            elif line.startswith("+++ b/"):
                paths.append(line[6:].split("\t", 1)[0].replace("\\", "/"))
            # Permit ordinary 0644 new text files, but reject mode changes,
            # executable/symlink/submodule modes, renames/copies, and binary patches.
            if line.startswith(("old mode ", "new mode ", "deleted file mode ",
                                "rename from ", "rename to ", "copy from ", "copy to ",
                                "GIT binary patch", "Binary files ")):
                raise SelfDeveloperError("File mode changes, rename/copy, and binary patches are not allowed")
            if line.startswith("new file mode ") and line != "new file mode 100644":
                raise SelfDeveloperError("Only regular non-executable text files may be added")
        # When Git file headers are present, require them to agree with the
        # unified-diff file markers. This prevents ambiguous/mismatched path pairs.
        for index, line in enumerate(lines):
            if not line.startswith("diff --git "):
                continue
            match = re.match(r"^diff --git a/(.*?) b/(.*?)$", line)
            if not match:
                raise SelfDeveloperError("Malformed or unsupported git diff header")
            section = []
            for candidate in lines[index + 1:]:
                if candidate.startswith("diff --git "):
                    break
                section.append(candidate)
            old_markers = [item[4:].split("\t", 1)[0] for item in section
                           if item.startswith("--- ")]
            new_markers = [item[4:].split("\t", 1)[0] for item in section
                           if item.startswith("+++ ")]
            if len(old_markers) != 1 or len(new_markers) != 1:
                raise SelfDeveloperError("Git diff headers require one matching old/new file marker pair")
            if old_markers[0] not in {f"a/{match.group(1)}", "/dev/null"}:
                raise SelfDeveloperError("Git diff header does not match its old file marker")
            if new_markers[0] not in {f"b/{match.group(2)}", "/dev/null"}:
                raise SelfDeveloperError("Git diff header does not match its new file marker")
        if not paths or not any(line.startswith("@@") for line in lines):
            raise SelfDeveloperError("Output is not a recognizable unified diff")
        protected = [item.lower() for item in PROTECTED_PATHS]
        for path in paths:
            if path == "/dev/null":
                continue
            if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
                raise SelfDeveloperError(f"Absolute paths are not allowed: {path}")
            if any(part in {"", ".", ".."} for part in path.split("/")):
                raise SelfDeveloperError(f"Unsafe patch path: {path}")
            normalized = path.lower()
            if any(normalized == item.rstrip("/") or normalized.startswith(item) for item in protected):
                raise SelfDeveloperError(f"Patch targets a protected path: {path}")
            if ".git" in normalized.split("/") or normalized.startswith(".github/workflows/") or normalized in {
                ".env", ".env.example", "requirements.txt", "pyproject.toml", "setup.py",
            }:
                raise SelfDeveloperError(f"Patch targets a restricted path: {path}")

    def _write_report(self, record: dict[str, Any]) -> dict[str, Any]:
        self.runs_root.mkdir(parents=True, exist_ok=True)
        run_id = record.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = self.runs_root / f"{run_id}.json"
        # Persist the same report path that the caller sees in the returned record.
        record["report_path"] = str(destination)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(destination)
        return record

    def latest_report(self) -> dict[str, Any]:
        reports = sorted(self.runs_root.glob("*.json"), reverse=True) if self.runs_root.exists() else []
        if not reports:
            raise FileNotFoundError("No self-developer reports found yet.")
        return json.loads(reports[0].read_text(encoding="utf-8"))
