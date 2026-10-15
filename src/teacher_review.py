"""Export teacher candidates for human review and import explicit CSV decisions.

Only rows marked 'approve' by a named reviewer are emitted to the curated dataset.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any
from src.teacher_dataset_audit import audit_teacher_dataset

FIELDS = ("fingerprint", "domain", "teacher_model", "instruction", "answer", "decision", "reviewer", "notes")

def _validate_dataset(source: Path) -> None:
    audit = audit_teacher_dataset(source)
    if not audit.passed:
        raise ValueError("teacher dataset has structural errors; fix/audit it before review")
    if audit.duplicate_fingerprints or audit.duplicate_instruction_answers:
        raise ValueError("teacher dataset contains duplicates; deduplicate before human review")

def export_review_queue(dataset: str | Path, csv_path: str | Path) -> int:
    source, destination = Path(dataset), Path(csv_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite review queue: {destination}")
    _validate_dataset(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open("r", encoding="utf-8") as stream, destination.open("x", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS); writer.writeheader()
        for line in stream:
            if not line.strip(): continue
            item = json.loads(line)
            if item["review_status"] != "unverified_candidate": continue
            writer.writerow({"fingerprint": item["fingerprint"], "domain": item["domain"],
                "teacher_model": item["teacher_model"], "instruction": item["instruction"],
                "answer": item["answer"], "decision": "", "reviewer": "", "notes": ""})
            count += 1
    return count

def import_review_decisions(dataset: str | Path, review_csv: str | Path, output: str | Path) -> dict[str, int]:
    source, decisions_path, destination = Path(dataset), Path(review_csv), Path(output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite curated dataset: {destination}")
    _validate_dataset(source)
    original: dict[str, dict[str, Any]] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item["review_status"] == "unverified_candidate": original[item["fingerprint"]] = item
    approved: dict[str, tuple[str, str]] = {}; rejected: set[str] = set()
    with decisions_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"review CSV must include columns: {', '.join(FIELDS)}")
        seen: set[str] = set()
        for line_number, row in enumerate(reader, 2):
            fingerprint = (row.get("fingerprint") or "").strip()
            if not fingerprint: raise ValueError(f"line {line_number}: fingerprint is required")
            if fingerprint in seen: raise ValueError(f"line {line_number}: duplicate review decision for {fingerprint}")
            seen.add(fingerprint)
            item = original.get(fingerprint)
            if item is None: raise ValueError(f"line {line_number}: unknown or already-reviewed fingerprint {fingerprint}")
            if any(row.get(key) != item[key] for key in ("domain", "instruction", "answer", "teacher_model")):
                raise ValueError(f"line {line_number}: content changed; re-export queue and review the original candidate")
            decision, reviewer = (row.get("decision") or "").strip().casefold(), (row.get("reviewer") or "").strip()
            if not decision:
                continue
            if decision not in {"approve", "reject"}:
                raise ValueError(f"line {line_number}: decision must be approve or reject")
            if not reviewer: raise ValueError(f"line {line_number}: reviewer is required")
            if decision == "approve": approved[fingerprint] = (reviewer, (row.get("notes") or "").strip())
            else: rejected.add(fingerprint)
    missing = set(original) - set(approved) - rejected
    if missing: raise ValueError(f"review incomplete: {len(missing)} candidate(s) have no explicit decision")
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts = {"approved": 0, "rejected": len(rejected)}
    with destination.open("x", encoding="utf-8", newline="\n") as out:
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            item = json.loads(line)
            if item["fingerprint"] not in approved: continue
            reviewer, notes = approved[item["fingerprint"]]
            item["review_status"] = "approved"
            item["review"] = {"reviewer": reviewer, "notes": notes, "decision": "approve"}
            out.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            counts["approved"] += 1
    counts["total_candidates"] = len(original)
    return counts

def main() -> None:
    parser = argparse.ArgumentParser(description="Export/import a human review queue for teacher-generated examples")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="export unverified candidates to CSV")
    export.add_argument("dataset"); export.add_argument("review_csv")
    ingest = sub.add_parser("import", help="import completed decisions and emit approved JSONL only")
    ingest.add_argument("dataset"); ingest.add_argument("review_csv"); ingest.add_argument("approved_jsonl")
    args = parser.parse_args()
    if args.command == "export":
        print(json.dumps({"exported_for_review": export_review_queue(args.dataset, args.review_csv)}, indent=2))
    else:
        print(json.dumps(import_review_decisions(args.dataset, args.review_csv, args.approved_jsonl), indent=2))

if __name__ == "__main__":
    main()
