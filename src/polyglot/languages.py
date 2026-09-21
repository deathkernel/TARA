"""Language capability metadata used by TARA's algorithm planner."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    executable: str
    strengths: frozenset[str]
    file_extension: str


def default_languages() -> tuple[LanguageSpec, ...]:
    return (
        LanguageSpec("python", "python", frozenset({"research", "prototyping", "data", "ai"}), ".py"),
        LanguageSpec("rust", "rustc", frozenset({"systems", "memory", "performance", "concurrency"}), ".rs"),
        LanguageSpec("cpp", "g++", frozenset({"performance", "algorithms", "systems"}), ".cpp"),
        LanguageSpec("go", "go", frozenset({"concurrency", "services", "systems"}), ".go"),
        LanguageSpec("java", "java", frozenset({"portability", "backend", "jvm"}), ".java"),
    )


def select_language(problem: str, languages: tuple[LanguageSpec, ...] | None = None) -> LanguageSpec:
    """Select a language deterministically from problem capability keywords.

    This is a planning heuristic, not a claim that one language is universally
    optimal. Benchmark results should be allowed to override it later.
    """
    text = problem.lower()
    languages = languages or default_languages()
    if any(word in text for word in ("memory", "systems", "low level", "embedded")):
        return next(x for x in languages if x.name == "rust")
    if any(word in text for word in ("performance", "fast", "competitive programming", "algorithm")):
        return next(x for x in languages if x.name == "cpp")
    if any(word in text for word in ("concurrent", "concurrency", "service", "server")):
        return next(x for x in languages if x.name == "go")
    return next(x for x in languages if x.name == "python")
