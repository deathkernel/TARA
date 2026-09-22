"""Evidence-based diagnosis of real-model capability benchmark failures."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable

from .intelligence_benchmark import BenchmarkCase
from .model_capability_runner import ModelEvaluation


@dataclass(frozen=True)
class CapabilityFailure:
    case_id: str
    category: str
    prompt: str
    expected: Any
    actual: str
    score: float
    kind: str
    severity: float
    training_hint: str


@dataclass(frozen=True)
class TrainingTarget:
    category: str
    priority: float
    failed_cases: int
    mean_score: float
    case_ids: tuple[str, ...]
    training_hint: str


@dataclass(frozen=True)
class FailureDiagnosisReport:
    checkpoint: str
    benchmark: str
    failures: tuple[CapabilityFailure, ...]
    targets: tuple[TrainingTarget, ...]
    fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "benchmark": self.benchmark,
            "failures": [asdict(item) for item in self.failures],
            "targets": [asdict(item) for item in self.targets],
            "fingerprint": self.fingerprint,
        }


class CapabilityFailureAnalyzer:
    """Turn measured outputs into bounded, deterministic training targets.

    Diagnosis is deliberately descriptive: it records what the model emitted,
    how the benchmark scored it, and a conservative training hint. It does not
    infer hidden causes from a single failed answer.
    """

    def __init__(self, *, max_targets: int = 7):
        if max_targets <= 0:
            raise ValueError("max_targets must be positive")
        self.max_targets = max_targets

    @staticmethod
    def _kind(actual: str, expected: Any, score: float) -> tuple[str, str]:
        text = actual.strip()
        if not text:
            return "empty_output", "Increase supervised examples that demonstrate the expected response format."
        if len(text) > 512:
            return "overlong_output", "Add concise target-response examples and enforce bounded response formatting."
        if text.lower() in {"i don't know", "unknown", "none", "n/a"}:
            return "non_answer", "Add grounded examples that map the task pattern to a direct answer."
        if score <= 0.0:
            expected_text = str(expected).strip().casefold()
            actual_text = text.casefold()
            if expected_text and expected_text in actual_text:
                return "format_mismatch", "Add formatting-normalization examples and preserve the required answer form."
            return "incorrect_answer", "Add diverse supervised examples for this capability and verify them with deterministic tests."
        return "partial", "Add examples targeting the missing details while retaining regression tests."

    def diagnose(
        self,
        evaluation: ModelEvaluation,
        cases: Iterable[BenchmarkCase],
    ) -> FailureDiagnosisReport:
        case_map = {case.case_id: case for case in cases}
        failures: list[CapabilityFailure] = []
        for result in evaluation.benchmark.results:
            if result.passed:
                continue
            case = case_map.get(result.case_id)
            if case is None:
                continue
            actual = evaluation.observations.get(result.case_id, "")
            kind, hint = self._kind(actual, case.expected, result.score)
            severity = round(1.0 - result.score, 6)
            failures.append(CapabilityFailure(
                case_id=case.case_id,
                category=case.category,
                prompt=case.prompt,
                expected=case.expected,
                actual=actual,
                score=result.score,
                kind=kind,
                severity=severity,
                training_hint=hint,
            ))

        grouped: dict[str, list[CapabilityFailure]] = {}
        for failure in failures:
            grouped.setdefault(failure.category, []).append(failure)
        targets: list[TrainingTarget] = []
        for category, items in grouped.items():
            mean_score = sum(item.score for item in items) / len(items)
            priority = round((1.0 - mean_score) * len(items), 6)
            hints = sorted({item.training_hint for item in items})
            targets.append(TrainingTarget(
                category=category,
                priority=priority,
                failed_cases=len(items),
                mean_score=round(mean_score, 6),
                case_ids=tuple(sorted(item.case_id for item in items)),
                training_hint=hints[0] if len(hints) == 1 else "Combine targeted examples for the observed failure modes and retain deterministic regression tests.",
            ))
        targets.sort(key=lambda item: (-item.priority, item.category))
        targets = targets[: self.max_targets]
        payload = {
            "checkpoint": evaluation.checkpoint,
            "benchmark": evaluation.benchmark.name,
            "failures": [asdict(item) for item in failures],
            "targets": [asdict(item) for item in targets],
        }
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return FailureDiagnosisReport(evaluation.checkpoint, evaluation.benchmark.name, tuple(failures), tuple(targets), fingerprint)


def write_diagnosis(report: FailureDiagnosisReport, path: str) -> None:
    from pathlib import Path
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
