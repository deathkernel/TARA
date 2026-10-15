"""End-to-end, resumable orchestration for TARA's teacher-data lifecycle.

The workflow deliberately pauses for human review between prepare and curate.
Each stage writes new artifacts and refuses to overwrite existing user data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.teacher_dataset_audit import audit_teacher_dataset
from src.teacher_distillation import DOMAIN_TASKS, generate_teacher_dataset
from src.teacher_review import export_review_queue, import_review_decisions
from src.teacher_training import train_curated_candidate, validate_approved_dataset


def _distinct_new_paths(*named_paths: tuple[str, str | Path]) -> dict[str, Path]:
    resolved = {name: Path(value).expanduser().resolve() for name, value in named_paths}
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("all pipeline artifact paths must be distinct")
    for name, path in resolved.items():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {name}: {path}")
        if name == "candidate dataset" and path.with_name(path.name + ".tmp").exists():
            raise FileExistsError(f"Refusing to overwrite generator temporary file: {path.with_name(path.name + '.tmp')}")
    return resolved


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def prepare_pipeline(
    provider: Any,
    candidate_dataset: str | Path,
    audit_report: str | Path,
    review_csv: str | Path,
    manifest_path: str | Path,
    *,
    per_domain: int = 2,
    max_completion_tokens: int = 256,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Generate candidate data, audit it, and create a CSV for human review."""
    if per_domain <= 0 or max_completion_tokens <= 0:
        raise ValueError("per_domain and max_completion_tokens must be positive")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0 and 2")
    paths = _distinct_new_paths(
        ("candidate dataset", candidate_dataset),
        ("audit report", audit_report),
        ("review CSV", review_csv),
        ("prepare manifest", manifest_path),
    )
    counts = generate_teacher_dataset(
        provider, paths["candidate dataset"], per_domain=per_domain,
        max_completion_tokens=max_completion_tokens, temperature=temperature, overwrite=False,
    )
    audit = audit_teacher_dataset(paths["candidate dataset"])
    audit_payload = audit.as_dict()
    _write_new_json(paths["audit report"], audit_payload)
    if not audit.passed:
        raise ValueError(f"teacher dataset audit failed; inspect {paths['audit report']}")
    if audit.duplicate_fingerprints or audit.duplicate_instruction_answers:
        raise ValueError(f"teacher dataset contains duplicates; inspect {paths['audit report']}")
    exported = export_review_queue(paths["candidate dataset"], paths["review CSV"])
    if exported != counts["total"]:
        raise ValueError(f"review queue count mismatch: generated={counts['total']}, exported={exported}")
    manifest = {
        "schema_version": 1,
        "stage": "awaiting_human_review",
        "teacher_model": getattr(provider, "model", "injected-provider"),
        "domain_count": len(DOMAIN_TASKS),
        "generation_counts": counts,
        "dataset_sha256": hashlib.sha256(paths["candidate dataset"].read_bytes()).hexdigest(),
        "audit_fingerprint": audit.fingerprint,
        "artifacts": {key: str(value) for key, value in paths.items() if key != "prepare manifest"},
        "next_action": "Review every CSV row independently, fill approve/reject and reviewer, then run the curate command.",
        "warning": "Structural/provenance checks do not prove answers are factually or mathematically correct.",
    }
    _write_new_json(paths["prepare manifest"], manifest)
    return manifest


def curate_pipeline(
    candidate_dataset: str | Path,
    review_csv: str | Path,
    approved_dataset: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Import explicit human decisions and emit only approved records."""
    paths = _distinct_new_paths(
        ("candidate dataset", candidate_dataset),
        ("review CSV", review_csv),
        ("approved dataset", approved_dataset),
        ("curation manifest", manifest_path),
    )
    counts = import_review_decisions(
        paths["candidate dataset"], paths["review CSV"], paths["approved dataset"]
    )
    approved_info = validate_approved_dataset(paths["approved dataset"])
    manifest = {
        "schema_version": 1,
        "stage": "approved_dataset_ready",
        "review_counts": counts,
        "approved_dataset": approved_info,
        "review_csv_sha256": hashlib.sha256(paths["review CSV"].read_bytes()).hexdigest(),
        "next_action": "Train a separate candidate checkpoint using the train command; keep the baseline unchanged.",
    }
    _write_new_json(paths["curation manifest"], manifest)
    return manifest


def train_pipeline(
    approved_dataset: str | Path,
    baseline_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    evidence_report: str | Path,
    *,
    steps: int = 100,
    device: str | None = None,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    """Train a new candidate from approved data and persist held-out evidence."""
    return train_curated_candidate(
        approved_dataset, baseline_checkpoint, candidate_checkpoint, evidence_report,
        steps=steps, device=device, max_new_tokens=max_new_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TARA end-to-end teacher pipeline: prepare -> human review -> curate -> train"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="generate candidates, audit them, and export a human review CSV")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--audit-report", required=True)
    prepare.add_argument("--review-csv", required=True)
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--per-domain", type=int, default=2)
    prepare.add_argument("--max-tokens", type=int, default=256)
    prepare.add_argument("--temperature", type=float, default=0.2)

    curate = sub.add_parser("curate", help="import explicit CSV decisions and write approved-only JSONL")
    curate.add_argument("--dataset", required=True)
    curate.add_argument("--review-csv", required=True)
    curate.add_argument("--approved-dataset", required=True)
    curate.add_argument("--manifest", required=True)

    train = sub.add_parser("train", help="train and evaluate a separate TARA candidate from approved JSONL")
    train.add_argument("--dataset", required=True)
    train.add_argument("--baseline", required=True)
    train.add_argument("--candidate", required=True)
    train.add_argument("--report", required=True)
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--device", default=None)
    train.add_argument("--max-new-tokens", type=int, default=32)

    args = parser.parse_args()
    if args.command == "prepare":
        os.environ.setdefault("TARA_LLM_BACKEND", "ollama")
        from src.llm_provider_factory import provider_from_env
        provider = provider_from_env()
        if provider is None:
            raise SystemExit("No teacher provider configured. Set TARA_LLM_BACKEND=ollama and check ollama list.")
        result = prepare_pipeline(
            provider, args.dataset, args.audit_report, args.review_csv, args.manifest,
            per_domain=args.per_domain, max_completion_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    elif args.command == "curate":
        result = curate_pipeline(args.dataset, args.review_csv, args.approved_dataset, args.manifest)
    else:
        result = train_pipeline(
            args.dataset, args.baseline, args.candidate, args.report,
            steps=args.steps, device=args.device, max_new_tokens=args.max_new_tokens,
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, default=str))
    if args.command == "train" and not result["decision"]["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
