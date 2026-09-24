"""Disabled host-process execution for generated programs.

Generated source must not be compiled or executed directly on the user's host.
This module keeps the result/API types used by the rest of TARA, but fails
closed before creating any process. Use a dedicated container/VM service for
untrusted execution outside this repository.
"""

from __future__ import annotations

from .candidate import PolyglotCandidate
from .languages import LanguageSpec, default_languages
from .result import ExecutionResult


class PolyglotExecutionError(ValueError):
    """Raised when an execution request is invalid or disabled."""


class PolyglotExecutor:
    """Fail-closed executor: no host compiler/interpreter process is started."""

    def __init__(
        self,
        languages: tuple[LanguageSpec, ...] | None = None,
        compile_timeout: float = 20.0,
        run_timeout: float = 5.0,
        output_limit: int = 64 * 1024,
        memory_limit_mb: int = 512,
    ) -> None:
        self.languages = languages or default_languages()
        self.compile_timeout = compile_timeout
        self.run_timeout = run_timeout
        self.output_limit = output_limit
        self.memory_limit_mb = memory_limit_mb

    def execute(self, candidate: PolyglotCandidate, stdin: str = "") -> ExecutionResult:
        """Return a safe failure without launching any host process."""
        spec = self._language(candidate.language)
        return ExecutionResult(
            language=spec.name,
            success=False,
            phase="disabled",
            exit_code=None,
            stdout="",
            stderr="",
            duration_ms=0.0,
            timed_out=False,
            error="host process execution is disabled for TARA safety",
        )

    def _language(self, name: str) -> LanguageSpec:
        normalized = name.lower().strip()
        aliases = {"c++": "cpp", "golang": "go", "py": "python", "rs": "rust"}
        normalized = aliases.get(normalized, normalized)
        for spec in self.languages:
            if spec.name == normalized:
                return spec
        raise PolyglotExecutionError(f"Unknown language: {name}")
