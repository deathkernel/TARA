"""Execution-free structural similarity analysis for algorithm candidates.

This module deliberately reports novelty relative to TARA's archive only. It
must not be interpreted as a proof of research novelty or prior-art absence.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from .archive import CandidateArchive
from .candidate import PolyglotCandidate


@dataclass(frozen=True)
class SimilarityMatch:
    fingerprint: str
    language: str
    similarity: float
    verified: bool
    source_preview: str


@dataclass(frozen=True)
class NoveltyReport:
    candidate_fingerprint: str
    archive_size: int
    exact_match: bool
    nearest: tuple[SimilarityMatch, ...]
    new_to_archive: bool


def _python_structure(source: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    parts: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            # Literal values often obscure algorithmic structure.
            parts.append(type(node.value).__name__)
        elif isinstance(node, ast.Name):
            parts.append("NAME")
        elif isinstance(node, ast.arg):
            parts.append("ARG")
        elif isinstance(node, ast.Attribute):
            parts.append("ATTR")
        else:
            parts.append(type(node).__name__)
    return " ".join(parts)


def _generic_tokens(source: str) -> str:
    # Keep identifiers abstract while retaining operators and punctuation.
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|==|!=|<=|>=|&&|\|\||\+\+|--|[{}()\[\];,:.+*/%<>=!-]", source)
    keywords = {
        "if", "else", "elif", "for", "while", "return", "class", "public",
        "private", "static", "void", "int", "long", "float", "double",
        "fn", "let", "mut", "func", "package", "import", "from", "def",
        "true", "false", "null", "nil", "new", "const", "var", "range",
    }
    normalized = [token if token in keywords or not token[0].isalpha() else "NAME" for token in tokens]
    return " ".join(normalized)


def structural_signature(candidate: PolyglotCandidate) -> str:
    """Return an execution-free representation emphasizing code structure."""
    if candidate.language == "python":
        signature = _python_structure(candidate.source)
        if signature:
            return signature
    return _generic_tokens(candidate.source)


def similarity(left: PolyglotCandidate, right: PolyglotCandidate) -> float:
    """Compute a structural similarity in [0, 1]."""
    if left.language == right.language and left.source.strip() == right.source.strip():
        return 1.0
    return SequenceMatcher(None, structural_signature(left), structural_signature(right)).ratio()


class NoveltyAnalyzer:
    """Compare a candidate against archived candidates without executing code."""

    def __init__(self, archive: CandidateArchive, *, threshold: float = 0.80) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.archive = archive
        self.threshold = threshold

    def analyze(self, candidate: PolyglotCandidate, *, limit: int = 5) -> NoveltyReport:
        if limit < 1:
            raise ValueError("limit must be positive")
        fingerprint = self.archive.fingerprint(candidate)
        records = self.archive.history(candidate.problem)
        matches: list[SimilarityMatch] = []
        exact = False
        for record in records:
            archived = PolyglotCandidate(**record["candidate"])
            score = similarity(candidate, archived)
            if record["fingerprint"] == fingerprint:
                exact = True
            if score >= self.threshold or record["fingerprint"] == fingerprint:
                matches.append(
                    SimilarityMatch(
                        fingerprint=record["fingerprint"],
                        language=archived.language,
                        similarity=score,
                        verified=bool(record["benchmark"].get("verified", False)),
                        source_preview=archived.source[:160],
                    )
                )
        matches.sort(key=lambda item: (-item.similarity, not item.verified, item.fingerprint))
        return NoveltyReport(
            candidate_fingerprint=fingerprint,
            archive_size=len(records),
            exact_match=exact,
            nearest=tuple(matches[:limit]),
            new_to_archive=not exact and not matches,
        )
