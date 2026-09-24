"""Disabled host-terminal adapter for TARA.

TARA must not execute arbitrary host commands. This module keeps the historical
API so callers fail closed instead of gaining access to the user's PC.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
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
    """Fail-closed terminal interface; host command execution is disabled."""

    def __init__(
        self,
        allowed_commands: Iterable[str] = (),
        *,
        working_root: str | Path = ".",
        timeout: float = 5.0,
        output_limit: int = 65536,
    ) -> None:
        self.allowed_commands = tuple(sorted(set(allowed_commands)))
        self.working_root = Path(working_root).expanduser().resolve()
        self.timeout, self.output_limit = timeout, output_limit

    def run(self, command: Iterable[str]) -> TerminalResult:
        argv = tuple(str(x) for x in command)
        return TerminalResult(
            False,
            argv,
            error="host terminal execution is disabled for TARA safety",
        )
