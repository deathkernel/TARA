"""Strict structural/provenance audit for Llama teacher-generated JSONL.

A clean report establishes schema and consistency only; it does not certify that
teacher answers are correct. This module never edits the input dataset.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from src.teacher_distillation import DOMAIN_TASKS

@dataclass(frozen=True)
class AuditIssue:
    code: str
    severity: str
    line: int | None
    detail: str

@dataclass(frozen=True)
class TeacherDatasetAudit:
    path: str
    records: int
    valid_records: int
    duplicate_fingerprints: int
    duplicate_instruction_answers: int
    domain_counts: dict[str, int]
    fingerprint: str
    issues: tuple[AuditIssue, ...]

    @property
    def passed(self) -> bool:
        return not any(item.severity == "error" for item in self.issues)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "issues": [asdict(issue) for issue in self.issues],
                "passed": self.passed,
                "interpretation": "Structural/provenance audit only. Passing does not establish factual, mathematical, or code correctness; human review and held-out evaluation remain required."}

def _fingerprint(domain: str, instruction: str, answer: str) -> str:
    return hashlib.sha256(f"{domain}\n{instruction}\n{answer}".encode("utf-8")).hexdigest()

def audit_teacher_dataset(path: str | Path) -> TeacherDatasetAudit:
    source = Path(path)
    issues: list[AuditIssue] = []
    records: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    pairs: list[str] = []
    domain_counts: Counter[str] = Counter()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FileNotFoundError(f"cannot read teacher dataset {source}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            issues.append(AuditIssue("blank-line", "warning", line_number, "blank JSONL line ignored"))
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(AuditIssue("malformed-json", "error", line_number, str(exc)))
            continue
        if not isinstance(item, dict):
            issues.append(AuditIssue("record-not-object", "error", line_number, "JSONL record must be an object"))
            continue
        records.append(item)
        required = ("text", "domain", "instruction", "answer", "source", "teacher_model", "fingerprint", "review_status")
        missing = [key for key in required if key not in item]
        if missing:
            issues.append(AuditIssue("missing-fields", "error", line_number, f"missing fields: {', '.join(missing)}"))
            continue
        invalid_types = [key for key in required if not isinstance(item.get(key), str)]
        if invalid_types:
            issues.append(AuditIssue("invalid-field-types", "error", line_number, f"fields must be strings: {', '.join(invalid_types)}"))
            continue
        domain, instruction, answer = item["domain"], item["instruction"], item["answer"]
        if domain not in DOMAIN_TASKS:
            issues.append(AuditIssue("unknown-domain", "error", line_number, f"unsupported domain: {domain!r}"))
        else:
            domain_counts[domain] += 1
        if not instruction.strip() or not answer.strip():
            issues.append(AuditIssue("empty-content", "error", line_number, "instruction and answer must be non-empty"))
        if item["source"] != "ollama_teacher_generated":
            issues.append(AuditIssue("unexpected-source", "warning", line_number, f"source is {item['source']!r}"))
        if not item["teacher_model"].strip():
            issues.append(AuditIssue("missing-teacher-model", "error", line_number, "teacher_model must be non-empty"))
        if item["review_status"] not in {"unverified_candidate", "approved", "rejected"}:
            issues.append(AuditIssue("invalid-review-status", "error", line_number, f"unsupported review_status: {item['review_status']!r}"))
        if item["fingerprint"] != _fingerprint(domain, instruction, answer):
            issues.append(AuditIssue("fingerprint-mismatch", "error", line_number, "fingerprint does not match domain/instruction/answer"))
        if item["text"] != f"Domain: {domain}\nInstruction: {instruction}\nAnswer: {answer}":
            issues.append(AuditIssue("text-field-mismatch", "error", line_number, "text field differs from canonical domain/instruction/answer rendering"))
        fingerprints.append(item["fingerprint"])
        pairs.append(hashlib.sha256(f"{domain}\n{instruction.strip().casefold()}\n{answer.strip().casefold()}".encode()).hexdigest())
        if not any(issue.severity == "error" and issue.line == line_number for issue in issues):
            valid_record_count += 1
    fp_counts, pair_counts = Counter(fingerprints), Counter(pairs)
    duplicate_fingerprints = sum(count - 1 for count in fp_counts.values() if count > 1)
    duplicate_pairs = sum(count - 1 for count in pair_counts.values() if count > 1)
    if duplicate_fingerprints:
        issues.append(AuditIssue("duplicate-fingerprint", "warning", None, f"{duplicate_fingerprints} repeated fingerprint(s) found"))
    if duplicate_pairs:
        issues.append(AuditIssue("duplicate-instruction-answer", "warning", None, f"{duplicate_pairs} duplicate domain/instruction/answer pair(s) found"))
    corpus_fingerprint = hashlib.sha256(json.dumps(sorted(fingerprints), separators=(",", ":")).encode()).hexdigest()
    return TeacherDatasetAudit(str(source), len(lines), valid_record_count, duplicate_fingerprints, duplicate_pairs,
                               {domain: domain_counts.get(domain, 0) for domain in DOMAIN_TASKS},
                               corpus_fingerprint, tuple(issues))

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit teacher JSONL structure, provenance, duplicates and fingerprints")
    parser.add_argument("dataset")
    parser.add_argument("--report", default=None, help="optional JSON report path; existing report is not overwritten")
    parser.add_argument("--fail-on-warnings", action="store_true")
    args = parser.parse_args()
    result = audit_teacher_dataset(args.dataset)
    payload = result.as_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.report:
        report_path = Path(args.report)
        if report_path.exists():
            raise SystemExit(f"Refusing to overwrite audit report: {report_path}")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result.passed or (args.fail_on_warnings and any(i.severity == "warning" for i in result.issues)):
        raise SystemExit(2)

if __name__ == "__main__":
    main()
