"""Common representation for algorithms written in different languages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PolyglotCandidate:
    problem: str
    language: str
    source: str
    metadata: dict[str, str] = field(default_factory=dict)
    verified: bool = False
    score: float | None = None
    feedback: str = ""

    @property
    def extension(self) -> str:
        return {
            "python": ".py",
            "rust": ".rs",
            "cpp": ".cpp",
            "go": ".go",
            "java": ".java",
        }.get(self.language, ".txt")
