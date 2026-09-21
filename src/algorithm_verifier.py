"""Deterministic verification helpers for algorithm candidates."""

from __future__ import annotations

import random
from typing import Any

from .sandbox_runner import run_candidate


def sorting_cases(seed: int = 42, count: int = 32) -> list[list[int]]:
    """Create deterministic edge and randomized sorting inputs."""
    cases: list[list[int]] = [[], [1], [2, 1], [1, 1, 1], [-3, 0, 2, -1], list(range(20, -1, -1))]
    rng = random.Random(seed)
    for _ in range(max(0, count - len(cases))):
        n = rng.randint(0, 40)
        cases.append([rng.randint(-100, 100) for _ in range(n)])
    return cases[:count]


def verify_sorting(source: str, *, timeout: float = 2.0) -> tuple[bool, float | None, str]:
    """Verify solve(value) returns the sorted version of an integer list.

    The score is the number of test cases executed successfully; lower is not
    inherently better here, so callers should treat the returned score as a
    coverage metric rather than a performance objective.
    """
    cases = sorting_cases()
    for index, case in enumerate(cases):
        result = run_candidate(source, case, timeout=timeout)
        if not result.ok:
            return False, float(index), f"case {index} failed: {result.error}"
        expected = sorted(case)
        if result.output != expected:
            return False, float(index), (
                f"counterexample case {index}: input={case!r}, "
                f"expected={expected!r}, got={result.output!r}"
            )
    return True, float(len(cases)), f"passed {len(cases)} deterministic cases"


def verify_problem(problem: str, source: str, *, timeout: float = 2.0) -> tuple[bool, float | None, str]:
    """Dispatch a known benchmark problem to its verifier."""
    normalized = problem.strip().lower()
    if normalized in {"sort an array of integers", "sort an array", "sorting"}:
        return verify_sorting(source, timeout=timeout)
    return False, None, f"no verifier registered for problem: {problem}"
