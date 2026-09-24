"""Controlled compile/run orchestration for TARA's generated programs.

This is defense-in-depth, not a security boundary for hostile code. For truly
untrusted candidates, run this layer inside a container or VM with OS limits.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .candidate import PolyglotCandidate
from .languages import LanguageSpec, default_languages
from .result import ExecutionResult


class PolyglotExecutionError(ValueError):
    """Raised when an execution request is invalid before a process starts."""


class PolyglotExecutor:
    """Compile and execute candidates with bounded time and captured output."""

    def __init__(
        self,
        languages: tuple[LanguageSpec, ...] | None = None,
        compile_timeout: float = 20.0,
        run_timeout: float = 5.0,
        output_limit: int = 64 * 1024,
        memory_limit_mb: int = 512,
    ) -> None:
        if compile_timeout <= 0 or run_timeout <= 0:
            raise ValueError("timeouts must be positive")
        if output_limit <= 0 or memory_limit_mb <= 0:
            raise ValueError("output and memory limits must be positive")
        self.languages = languages or default_languages()
        self.compile_timeout = compile_timeout
        self.run_timeout = run_timeout
        self.output_limit = output_limit
        self.memory_limit_mb = memory_limit_mb

    def execute(self, candidate: PolyglotCandidate, stdin: str = "") -> ExecutionResult:
        spec = self._language(candidate.language)
        with tempfile.TemporaryDirectory(prefix="tara_polyglot_") as temp:
            root = Path(temp)
            source = root / f"candidate{spec.file_extension}"
            source.write_text(candidate.source, encoding="utf-8")

            if spec.name == "python":
                command = [spec.executable, "-I", "-S", str(source)]
                compile_result = None
            elif spec.name == "rust":
                binary = root / "candidate_bin"
                compile_result = self._run(
                    [spec.executable, str(source), "-O", "-o", str(binary)],
                    "",
                    self.compile_timeout,
                )
                command = [str(binary)]
            elif spec.name == "cpp":
                binary = root / "candidate_bin"
                compile_result = self._run(
                    [spec.executable, "-std=c++17", "-O2", str(source), "-o", str(binary)],
                    "",
                    self.compile_timeout,
                )
                command = [str(binary)]
            elif spec.name == "go":
                binary = root / "candidate_bin"
                compile_result = self._run(
                    [spec.executable, "build", "-o", str(binary), str(source)],
                    "",
                    self.compile_timeout,
                )
                command = [str(binary)]
            elif spec.name == "java":
                compile_result = self._run(
                    ["javac", str(source)],
                    "",
                    self.compile_timeout,
                )
                command = [spec.executable, "-cp", str(root), "candidate"]
            else:
                raise PolyglotExecutionError(f"Unsupported language: {spec.name}")

            if compile_result is not None and compile_result[0] != 0:
                return ExecutionResult(
                    language=spec.name,
                    success=False,
                    phase="compile",
                    exit_code=compile_result[0],
                    stdout=compile_result[1],
                    stderr=compile_result[2],
                    duration_ms=compile_result[3],
                    timed_out=compile_result[4],
                    error="compilation failed",
                )

            exit_code, stdout, stderr, duration_ms, timed_out = self._run(
                command, stdin, self.run_timeout
            )
            return ExecutionResult(
                language=spec.name,
                success=exit_code == 0 and not timed_out,
                phase="run",
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_ms=duration_ms,
                timed_out=timed_out,
                error=None if exit_code == 0 and not timed_out else "execution failed",
            )

    def _language(self, name: str) -> LanguageSpec:
        normalized = name.lower().strip()
        aliases = {"c++": "cpp", "golang": "go", "py": "python", "rs": "rust"}
        normalized = aliases.get(normalized, normalized)
        for spec in self.languages:
            if spec.name == normalized:
                return spec
        raise PolyglotExecutionError(f"Unknown language: {name}")

    def _resource_limits(self, timeout: float):
        if os.name == "nt":
            return None
        import resource
        memory = self.memory_limit_mb * 1024 * 1024
        cpu = max(1, int(timeout) + 1)
        def apply_limits():
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        return apply_limits

    def _run(self, command: list[str], stdin: str, timeout: float, *, cwd: Path | None = None):
        env = {"PATH": os.environ.get("PATH", "")}
        start = time.perf_counter()
        try:
            process = subprocess.run(
                command,
                input=stdin,
                text=True,
                capture_output=True,
                timeout=timeout,
                shell=False,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                start_new_session=(os.name != "nt"),
                preexec_fn=self._resource_limits(timeout) if os.name != "nt" else None,
            )
            duration_ms = (time.perf_counter() - start) * 1000.0
            return (
                process.returncode,
                process.stdout[: self.output_limit],
                process.stderr[: self.output_limit],
                duration_ms,
                False,
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            stdout = (exc.stdout or "")
            stderr = (exc.stderr or "")
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return (
                None,
                stdout[: self.output_limit],
                stderr[: self.output_limit],
                duration_ms,
                True,
            )
        except OSError as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            return (None, "", str(exc)[: self.output_limit], duration_ms, False)
