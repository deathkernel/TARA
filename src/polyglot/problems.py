"""Problem contracts and deterministic benchmark cases for TARA."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TestCase:
    stdin: str
    expected_stdout: str


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    description: str
    tests: tuple[TestCase, ...]


SORTING = ProblemSpec(
    name="sorting",
    description="Read n followed by n integers and print them in nondecreasing order.",
    tests=(
        TestCase("5\n5 1 4 2 3\n", "1 2 3 4 5\n"),
        TestCase("6\n9 -1 9 0 3 -5\n", "-5 -1 0 3 9 9\n"),
        TestCase("1\n42\n", "42\n"),
        TestCase("0\n\n", "\n"),
    ),
)

BINARY_SEARCH = ProblemSpec(
    name="binary_search",
    description="Read n, n sorted integers, and a target; print its first zero-based index or -1.",
    tests=(
        TestCase("5\n1 3 5 7 9\n7\n", "3\n"),
        TestCase("5\n1 3 5 7 9\n2\n", "-1\n"),
        TestCase("6\n1 2 2 2 4 8\n2\n", "1\n"),
    ),
)


_PROBLEMS = {SORTING.name: SORTING, BINARY_SEARCH.name: BINARY_SEARCH}


def get_problem(name: str) -> ProblemSpec:
    try:
        return _PROBLEMS[name.strip().lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark problem: {name}") from exc


def list_problems() -> tuple[ProblemSpec, ...]:
    return tuple(_PROBLEMS.values())
