"""Deterministic capability benchmarking for TARA.

Phase 37 starts by measuring capability before changing the model.  Benchmark
cases are deliberately separated from execution: callers provide a solver,
while this module handles validation, category aggregation, baselines and
regression gates.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    category: str
    prompt: str
    expected: Any
    scorer: Callable[[Any, Any], float] | None = None
    weight: float = 1.0

    def __post_init__(self):
        if not self.case_id.strip() or not self.category.strip():
            raise ValueError("case_id and category must be non-empty")
        if not self.prompt.strip():
            raise ValueError("prompt must be non-empty")
        if self.weight <= 0 or not math.isfinite(float(self.weight)):
            raise ValueError("weight must be finite and positive")

    def score(self, actual: Any) -> float:
        scorer = self.scorer or exact_score
        value = float(scorer(actual, self.expected))
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("case scorer must return a finite score in [0, 1]")
        return value


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    category: str
    score: float
    passed: bool
    weight: float
    error: str | None = None


@dataclass(frozen=True)
class CategoryScore:
    category: str
    score: float
    cases: int
    passed: int


@dataclass(frozen=True)
class BenchmarkReport:
    name: str
    case_count: int
    passed_cases: int
    overall_score: float
    categories: tuple[CategoryScore, ...]
    results: tuple[CaseResult, ...]
    fingerprint: str

    @property
    def passed(self) -> bool:
        return self.passed_cases == self.case_count and self.case_count > 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "categories": [asdict(item) for item in self.categories],
            "results": [asdict(item) for item in self.results],
        }


@dataclass(frozen=True)
class RegressionGate:
    passed: bool
    overall_delta: float
    minimum_overall: float
    category_deltas: dict[str, float]
    regressions: tuple[str, ...]


class IntelligenceBenchmark:
    """Run a bounded benchmark against an injected model/solver."""

    def __init__(self, name: str, cases: Iterable[BenchmarkCase]):
        self.name = name
        self.cases = tuple(cases)
        if not self.cases:
            raise ValueError("benchmark requires at least one case")
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("benchmark case IDs must be unique")

    def run(self, solver: Callable[[str], Any]) -> BenchmarkReport:
        results: list[CaseResult] = []
        for case in self.cases:
            try:
                actual = solver(case.prompt)
                score = case.score(actual)
                results.append(CaseResult(case.case_id, case.category, score, score >= 1.0, case.weight))
            except Exception as exc:
                results.append(CaseResult(case.case_id, case.category, 0.0, False, case.weight, str(exc)))

        total_weight = sum(item.weight for item in self.cases)
        overall = sum(r.score * r.weight for r in results) / total_weight
        categories: list[CategoryScore] = []
        for category in sorted({case.category for case in self.cases}):
            subset = [r for r in results if r.category == category]
            weight = sum(r.weight for r in subset)
            score = sum(r.score * r.weight for r in subset) / weight
            categories.append(CategoryScore(category, score, len(subset), sum(r.passed for r in subset)))

        payload = {
            "name": self.name,
            "cases": [
                {"id": c.case_id, "category": c.category, "prompt": c.prompt, "expected": c.expected, "weight": c.weight}
                for c in self.cases
            ],
            "results": [asdict(r) for r in results],
        }
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        return BenchmarkReport(self.name, len(results), sum(r.passed for r in results), overall,
                               tuple(categories), tuple(results), fingerprint)


def exact_score(actual: Any, expected: Any) -> float:
    return 1.0 if actual == expected else 0.0


def normalized_text_score(actual: Any, expected: Any) -> float:
    """Exact normalized text comparison for deterministic text tasks."""
    left = " ".join(str(actual).strip().split()).casefold()
    right = " ".join(str(expected).strip().split()).casefold()
    return 1.0 if left == right else 0.0


def compare_regression(
    baseline: BenchmarkReport,
    candidate: BenchmarkReport,
    *,
    minimum_overall: float = 0.0,
    maximum_category_drop: float = 0.05,
) -> RegressionGate:
    """Reject broad or category-level regressions beyond explicit tolerances."""
    if baseline.name != candidate.name:
        raise ValueError("baseline and candidate benchmarks must have the same name")
    if minimum_overall < 0 or maximum_category_drop < 0:
        raise ValueError("regression tolerances must be non-negative")

    base_categories = {item.category: item.score for item in baseline.categories}
    candidate_categories = {item.category: item.score for item in candidate.categories}
    deltas = {name: candidate_categories[name] - score for name, score in base_categories.items() if name in candidate_categories}
    regressions = tuple(sorted(
        name for name, delta in deltas.items() if delta < -maximum_category_drop
    ))
    overall_delta = candidate.overall_score - baseline.overall_score
    passed = candidate.overall_score >= minimum_overall and not regressions
    return RegressionGate(passed, overall_delta, minimum_overall, deltas, regressions)


def benchmark_cases() -> tuple[BenchmarkCase, ...]:
    """Small smoke suite covering core capability categories."""
    return (
        BenchmarkCase("code-echo", "coding", "return 2 + 2", "4", normalized_text_score),
        BenchmarkCase("logic-true", "reasoning", "Is 7 greater than 3?", True, lambda a, e: 1.0 if bool(a) is e else 0.0),
        BenchmarkCase("memory-fact", "memory", "Recall token TARA-37", "TARA-37", normalized_text_score),
        BenchmarkCase("plan-order", "planning", "What comes first: plan or execute?", "plan", normalized_text_score),
    )
