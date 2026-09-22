"""Deterministic dataset quality and leakage audit for TARA.

The auditor treats dataset preparation as an experimental input: it reports
empty/short examples, duplicates, split overlap, suspicious metadata markers,
and stable fingerprints without modifying source data.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class DatasetAuditIssue:
    code: str
    detail: str
    severity: str = "warning"


@dataclass(frozen=True)
class DatasetAuditReport:
    records: int
    unique_records: int
    duplicate_records: int
    empty_records: int
    short_records: int
    train_validation_overlap: int
    train_test_overlap: int
    validation_test_overlap: int
    suspicious_markers: int
    fingerprint: str
    issues: tuple[DatasetAuditIssue, ...]

    @property
    def healthy(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def as_dict(self) -> dict:
        return asdict(self)


def _text(record: object) -> str:
    if isinstance(record, str):
        return re.sub(r"\s+", " ", record).strip()
    if isinstance(record, Mapping):
        for key in ("text", "content", "prompt"):
            value = record.get(key)
            if isinstance(value, str):
                return re.sub(r"\s+", " ", value).strip()
        if "problem" in record and "solution" in record:
            value = "Problem: " + str(record["problem"]) + " Approach: " + str(record["solution"])
            if "tests" in record:
                value += " Tests: " + str(record["tests"])
            return re.sub(r"\s+", " ", value).strip()
    return ""


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DatasetAuditor:
    """Audit one corpus or explicit train/validation/test partitions."""

    def __init__(self, *, min_length: int = 8, suspicious_markers: Sequence[str] | None = None):
        if min_length < 1:
            raise ValueError("min_length must be positive")
        self.min_length = min_length
        self.suspicious_markers = tuple(suspicious_markers or (
            "test answer", "expected answer", "ground truth", "unit test",
        ))

    def audit(
        self,
        records: Iterable[object],
        *,
        train: Iterable[object] | None = None,
        validation: Iterable[object] | None = None,
        test: Iterable[object] | None = None,
    ) -> DatasetAuditReport:
        all_records = list(records)
        texts = [_text(item) for item in all_records]
        keys = [_key(text) for text in texts if text]
        unique = set(keys)
        duplicate_records = len(keys) - len(unique)
        empty_records = sum(not text for text in texts)
        short_records = sum(0 < len(text) < self.min_length for text in texts)

        def partition_keys(items: Iterable[object] | None) -> set[str]:
            return {_key(text) for item in (items or []) if (text := _text(item))}

        train_keys = partition_keys(train)
        validation_keys = partition_keys(validation)
        test_keys = partition_keys(test)
        tv = len(train_keys & validation_keys)
        tt = len(train_keys & test_keys)
        vt = len(validation_keys & test_keys)
        suspicious = sum(
            any(marker.lower() in text.lower() for marker in self.suspicious_markers)
            for text in texts
        )

        issues: list[DatasetAuditIssue] = []
        if empty_records:
            issues.append(DatasetAuditIssue("empty-records", f"{empty_records} records contain no usable text"))
        if short_records:
            issues.append(DatasetAuditIssue("short-records", f"{short_records} records are shorter than {self.min_length} characters"))
        if duplicate_records:
            issues.append(DatasetAuditIssue("duplicates", f"{duplicate_records} duplicate records detected"))
        if tv or tt or vt:
            issues.append(DatasetAuditIssue(
                "split-overlap",
                f"train/validation={tv}, train/test={tt}, validation/test={vt}",
                "error",
            ))
        if suspicious:
            issues.append(DatasetAuditIssue(
                "suspicious-markers",
                f"{suspicious} records contain configured evaluation markers",
            ))

        fingerprint = hashlib.sha256(
            json.dumps(sorted(keys), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return DatasetAuditReport(
            records=len(all_records),
            unique_records=len(unique),
            duplicate_records=duplicate_records,
            empty_records=empty_records,
            short_records=short_records,
            train_validation_overlap=tv,
            train_test_overlap=tt,
            validation_test_overlap=vt,
            suspicious_markers=suspicious,
            fingerprint=fingerprint,
            issues=tuple(issues),
        )

    def audit_jsonl(self, path: str | Path) -> DatasetAuditReport:
        records = []
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
        return self.audit(records)
