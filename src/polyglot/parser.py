"""Parse language-model output into executable polyglot candidates."""

from __future__ import annotations

import re

from .candidate import PolyglotCandidate


class CandidateParseError(ValueError):
    """Raised when model output does not contain a usable code candidate."""


_ALIASES = {
    "py": "python",
    "python3": "python",
    "rs": "rust",
    "cpp": "cpp",
    "c++": "cpp",
    "cc": "cpp",
    "golang": "go",
    "java": "java",
}
_SUPPORTED = {"python", "rust", "cpp", "go", "java"}


def _normalize_language(value: str) -> str:
    value = value.strip().lower()
    value = _ALIASES.get(value, value)
    if value not in _SUPPORTED:
        raise CandidateParseError(f"unsupported language: {value or '<empty>'}")
    return value


def parse_candidate(output: str, problem: str) -> PolyglotCandidate:
    """Extract one fenced code block and its declared language from model output."""
    if not output or not output.strip():
        raise CandidateParseError("model output is empty")

    language_match = re.search(r"(?im)^\s*(?:language|lang)\s*:\s*([^\n]+)", output)
    fence_match = re.search(r"```\s*([A-Za-z0-9+#-]+)?\s*\n(.*?)```", output, re.DOTALL)
    if not fence_match:
        raise CandidateParseError("no fenced source code found")

    fence_language = fence_match.group(1) or ""
    declared = language_match.group(1) if language_match else fence_language
    if not declared:
        raise CandidateParseError("candidate language was not declared")

    language = _normalize_language(declared)
    source = fence_match.group(2).strip()
    if not source:
        raise CandidateParseError("source code is empty")

    return PolyglotCandidate(
        problem=problem,
        language=language,
        source=source,
        metadata={"parser": "structured-fence-v1"},
    )
