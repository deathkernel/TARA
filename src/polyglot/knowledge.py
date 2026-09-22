"""Turn verified benchmark results into durable training knowledge."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from .benchmark import BenchmarkResult
from .problems import get_problem


@dataclass(frozen=True)
class KnowledgeRecord:
    """A provenance-preserving training example derived from verification."""

    fingerprint: str
    problem: str
    language: str
    solution: str
    tests: str
    correctness: float
    runtime_ms: float
    verified: bool
    provenance: str = "polyglot-benchmark"


class KnowledgeExtractor:
    """Extract only verified candidates into a deduplicated JSONL dataset."""

    def __init__(self, *, verified_only: bool = True) -> None:
        self.verified_only = verified_only

    def from_result(self, result: BenchmarkResult) -> KnowledgeRecord | None:
        if self.verified_only and not result.verified:
            return None
        candidate = result.candidate
        spec = get_problem(candidate.problem)
        tests = "\n".join(
            f"INPUT={case.stdin!r} OUTPUT={case.expected_stdout!r}"
            for case in spec.tests
        )
        fingerprint = hashlib.sha256(
            f"{candidate.problem}\0{candidate.language}\0{candidate.source}".encode("utf-8")
        ).hexdigest()
        return KnowledgeRecord(
            fingerprint=fingerprint,
            problem=candidate.problem,
            language=candidate.language,
            solution=candidate.source,
            tests=tests,
            correctness=result.correctness,
            runtime_ms=result.total_runtime_ms,
            verified=result.verified,
        )

    def extract(self, results: Iterable[BenchmarkResult]) -> tuple[KnowledgeRecord, ...]:
        records: dict[str, KnowledgeRecord] = {}
        for result in results:
            record = self.from_result(result)
            if record is not None:
                records.setdefault(record.fingerprint, record)
        return tuple(records.values())

    def export_jsonl(
        self,
        records: Iterable[KnowledgeRecord],
        path: str | Path,
    ) -> int:
        """Write deterministic JSONL suitable for ``train_algorithm_lm.py``."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        records = tuple(records)
        with destination.open("w", encoding="utf-8") as handle:
            for record in records:
                payload = asdict(record)
                # Keep the existing trainer contract while preserving provenance.
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        return len(records)

    def export_archive(self, archive, path: str | Path, problem: str | None = None) -> int:
        """Export verified archive records without executing archived source."""
        results: list[BenchmarkResult] = []
        for item in archive.history(problem):
            candidate_data = item["candidate"]
            candidate = __import__(
                "src.polyglot.candidate", fromlist=["PolyglotCandidate"]
            ).PolyglotCandidate(**candidate_data)
            benchmark = item["benchmark"]
            results.append(
                BenchmarkResult(
                    candidate=candidate,
                    passed=int(benchmark["passed"]),
                    total=int(benchmark["total"]),
                    correctness=float(benchmark["correctness"]),
                    total_runtime_ms=float(benchmark["total_runtime_ms"]),
                    failures=tuple(benchmark.get("failures", ())),
                )
            )
        return self.export_jsonl(self.extract(results), path)
