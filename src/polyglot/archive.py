"""Persistent, execution-free memory for verified and rejected candidates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkResult
from .candidate import PolyglotCandidate


class CandidateArchive:
    """Append-only JSONL archive keyed by a deterministic candidate fingerprint.

    The archive stores source and benchmark metadata only. It never executes
    archived source code, making it suitable as a long-term research memory.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @staticmethod
    def fingerprint(candidate: PolyglotCandidate) -> str:
        payload = f"{candidate.problem}\0{candidate.language}\0{candidate.source}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def save(self, result: BenchmarkResult) -> bool:
        """Persist a result unless the exact candidate is already archived.

        Returns True when a new record was written.
        """
        fingerprint = self.fingerprint(result.candidate)
        existing = {item["fingerprint"] for item in self._records()}
        if fingerprint in existing:
            return False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "fingerprint": fingerprint,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "candidate": asdict(result.candidate),
            "benchmark": {
                "passed": result.passed,
                "total": result.total,
                "correctness": result.correctness,
                "total_runtime_ms": result.total_runtime_ms,
                "failures": list(result.failures),
                "verified": result.verified,
            },
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return True

    def history(self, problem: str | None = None) -> tuple[dict[str, Any], ...]:
        records = self._records()
        if problem is None:
            return tuple(records)
        normalized = problem.strip().lower()
        return tuple(item for item in records if item["candidate"]["problem"].strip().lower() == normalized)

    def best(self, problem: str) -> dict[str, Any] | None:
        records = self.history(problem)
        if not records:
            return None
        return min(
            records,
            key=lambda item: (
                -float(item["benchmark"]["correctness"]),
                float(item["benchmark"]["total_runtime_ms"]),
            ),
        )

    def stats(self, problem: str | None = None) -> dict[str, int]:
        records = self.history(problem)
        return {
            "candidates": len(records),
            "verified": sum(bool(item["benchmark"]["verified"]) for item in records),
            "languages": len({item["candidate"]["language"] for item in records}),
        }

    def _records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid archive JSON at line {line_number}") from exc
                if not isinstance(record, dict) or "fingerprint" not in record:
                    raise ValueError(f"invalid archive record at line {line_number}")
                records.append(record)
        return records
