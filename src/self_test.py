"""Controlled self-testing utilities for TARA.

Research basis:
- Automated regression suites are a standard software-engineering control for
  detecting unintended changes.
- Reproducible experiments require the exact command, exit status, and output
  to remain observable rather than replacing evidence with a simple boolean.

TARA's runner executes only the repository's declared pytest suite through the
current Python interpreter. It returns structured evidence and never executes
an arbitrary command supplied by a model.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys


@dataclass(frozen=True)
class TestRun:
    """Structured evidence from one test-suite execution."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def passed(self):
        return self.returncode == 0


def summarize_pytest_output(stdout, stderr=""):
    """Extract common pytest counts while preserving raw output separately."""
    text = "\n".join(part for part in (stdout, stderr) if part)
    summary = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    patterns = {
        "passed": r"(\d+) passed",
        "failed": r"(\d+) failed",
        "skipped": r"(\d+) skipped",
        "errors": r"(\d+) error(?:s)?",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            summary[name] = int(match.group(1))
    summary["output"] = text
    return summary


def run_test_suite(repo_root=".", timeout=300):
    """Run TARA's fixed pytest suite and return inspectable test evidence."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    root = Path(repo_root)
    if not root.is_dir():
        raise FileNotFoundError(str(root))

    command = (sys.executable, "-m", "pytest", "-q")
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return TestRun(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return TestRun(command, 124, str(stdout), str(stderr) + "\npytest timed out")
