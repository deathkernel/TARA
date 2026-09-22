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
    """Extract the final fenced program from model output.

    TARA's prompt contains a fenced format example. Selecting the final code
    fence prevents the prompt's placeholder from being mistaken for generated
    source when the runtime returns prompt + continuation.
    """
    if not output or not output.strip():
        raise CandidateParseError("model output is empty")

    language_match = re.search(r"(?im)^\s*(?:language|lang)\s*:\s*([^\n]+)", output)
    fence_matches = list(re.finditer(r"```\s*([A-Za-z0-9+#-]+)?\s*\n(.*?)```", output, re.DOTALL))
    if not fence_matches:
        raise CandidateParseError("no fenced source code found")

    fence_match = fence_matches[-1]
    fence_language = fence_match.group(1) or ""
    declared = fence_language or (language_match.group(1) if language_match else "")
    if not declared:
        raise CandidateParseError("candidate language was not declared")

    language = _normalize_language(declared)
    source = fence_match.group(2).strip()
    if not source or "<complete program>" in source:
        raise CandidateParseError("source code is empty or still contains the prompt placeholder")

    return PolyglotCandidate(
        problem=problem,
        language=language,
        source=source + "\n",
        metadata={"parser": "structured-fence-v2"},
    )
