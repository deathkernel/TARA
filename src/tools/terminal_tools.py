"""Basic constrained terminal tool for TARA.

Commands are allow-listed and run without a shell. This is a safety boundary,
not a secure sandbox for hostile code.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Iterable

@dataclass(frozen=True)
class TerminalResult:
    success: bool
    command: tuple[str, ...]
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    error: str | None = None

class TerminalTools:
    def __init__(self, allowed_commands: Iterable[str] = ("python", "python3"), *, working_root: str | Path = ".", timeout: float = 5.0, output_limit: int = 65536) -> None:
        commands = tuple(sorted(set(allowed_commands)))
        if not commands: raise ValueError("at least one command is required")
        if timeout <= 0 or output_limit < 1: raise ValueError("invalid limits")
        self.allowed_commands = commands
        self.working_root = Path(working_root).expanduser().resolve()
        self.timeout, self.output_limit = timeout, output_limit

    def run(self, command: Iterable[str]) -> TerminalResult:
        argv = tuple(str(x) for x in command)
        if not argv or argv[0] not in self.allowed_commands:
            return TerminalResult(False, argv, error="command is not allow-listed")
        try:
            proc = subprocess.run(argv, cwd=self.working_root, shell=False, capture_output=True, text=True, timeout=self.timeout, check=False)
            stdout, stderr = proc.stdout[:self.output_limit], proc.stderr[:self.output_limit]
            return TerminalResult(proc.returncode == 0, argv, stdout, stderr, proc.returncode)
        except subprocess.TimeoutExpired as exc:
            return TerminalResult(False, argv, error=f"command timed out after {self.timeout}s")
        except (OSError, ValueError) as exc:
            return TerminalResult(False, argv, error=f"{type(exc).__name__}: {exc}")
