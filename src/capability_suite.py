"""Expanded deterministic capability suite for TARA Phase 37.6.

The suite measures separate capabilities rather than collapsing everything into
one opaque score. Solvers are injected, so benchmarking does not grant tools or
execute arbitrary model output.
"""
from __future__ import annotations

from src.intelligence_benchmark import BenchmarkCase, IntelligenceBenchmark, normalized_text_score


def _bool_score(actual, expected):
    return 1.0 if isinstance(actual, bool) and actual is expected else 0.0


def _numeric_score(actual, expected):
    try:
        return 1.0 if float(actual) == float(expected) else 0.0
    except (TypeError, ValueError):
        return 0.0


def capability_cases() -> tuple[BenchmarkCase, ...]:
    """Return a compact, deterministic suite with multiple cases per capability."""
    return (
        BenchmarkCase("coding-arithmetic", "coding", "Compute 17 * 3", 51, _numeric_score),
        BenchmarkCase("coding-string", "coding", "Reverse the string 'tara'", "arat", normalized_text_score),
        BenchmarkCase("coding-order", "coding", "Sort 3, 1, 2 ascending", "1 2 3", normalized_text_score),
        BenchmarkCase("reasoning-transitive", "reasoning", "A is taller than B and B is taller than C. Is A taller than C?", True, _bool_score),
        BenchmarkCase("reasoning-negation", "reasoning", "If all birds are animals, are all animals birds?", False, _bool_score),
        BenchmarkCase("reasoning-count", "reasoning", "What is 12 / 3?", 4, _numeric_score),
        BenchmarkCase("memory-token", "memory", "Recall token TARA-37", "TARA-37", normalized_text_score),
        BenchmarkCase("memory-fact", "memory", "Recall that the project name is TARA", "TARA", normalized_text_score),
        BenchmarkCase("memory-order", "memory", "Recall sequence: perceive, reason, plan", "perceive reason plan", normalized_text_score),
        BenchmarkCase("planning-first", "planning", "Before executing a task, what should happen first?", "plan", normalized_text_score),
        BenchmarkCase("planning-dependency", "planning", "Task B depends on Task A. Which runs first?", "A", normalized_text_score),
        BenchmarkCase("planning-validation", "planning", "Should an unvalidated plan be executed immediately?", False, _bool_score),
        BenchmarkCase("tools-select", "tool_use", "Need to read a local file. Which capability is required?", "file-read", normalized_text_score),
        BenchmarkCase("tools-verify", "tool_use", "After a tool reports success, what should happen before trusting the result?", "verify", normalized_text_score),
        BenchmarkCase("tools-stop", "tool_use", "If the emergency stop is active, should a new tool action execute?", False, _bool_score),
        BenchmarkCase("algorithm-correctness", "algorithms", "What property must an algorithm satisfy before promotion?", "correctness", normalized_text_score),
        BenchmarkCase("algorithm-benchmark", "algorithms", "What should be compared when optimizing two verified algorithms?", "performance", normalized_text_score),
        BenchmarkCase("algorithm-regression", "algorithms", "If a candidate is faster but fails a required test, promote it?", False, _bool_score),
        BenchmarkCase("learning-replay", "learning", "What is replay used to reduce during continual learning?", "forgetting", normalized_text_score),
        BenchmarkCase("learning-evidence", "learning", "Should unverified knowledge enter the verified learning set?", False, _bool_score),
        BenchmarkCase("learning-improvement", "learning", "What should gate a claimed learning improvement?", "evidence", normalized_text_score),
    )


def capability_benchmark(name: str = "tara-capability-v1") -> IntelligenceBenchmark:
    return IntelligenceBenchmark(name, capability_cases())
