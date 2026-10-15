"""Train a new TARA candidate only from human-approved teacher data.

This command is explicit/offline: it never changes the teacher model or promotes
a checkpoint. The candidate is retained as a separate artifact and must pass the
same held-out capability gate before anyone considers promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from src.evaluation_orchestrator import CheckpointEvaluator
from src.teacher_dataset_audit import audit_teacher_dataset
from src.training_experiment import TrainingExperiment
from src.training_pipeline import TrainingConfig


def validate_approved_dataset(path: str | Path) -> dict[str, Any]:
    """Require a clean, unique dataset containing only explicitly reviewed rows."""
    source = Path(path).expanduser().resolve()
    audit = audit_teacher_dataset(source)
    if not audit.passed:
        raise ValueError("teacher dataset audit failed; training is blocked")
    if audit.duplicate_fingerprints or audit.duplicate_instruction_answers:
        raise ValueError("teacher dataset contains duplicates; training is blocked")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("review_status") != "approved":
            raise ValueError(f"line {line_number}: only review_status=approved records may train")
        review = item.get("review")
        if not isinstance(review, dict) or review.get("decision") != "approve" or not str(review.get("reviewer", "")).strip():
            raise ValueError(f"line {line_number}: explicit approval and reviewer are required")
        rows.append(item)
    if not rows:
        raise ValueError("approved teacher dataset is empty")
    return {
        "path": str(source),
        "records": len(rows),
        "audit_fingerprint": audit.fingerprint,
        "dataset_fingerprint": hashlib.sha256(source.read_bytes()).hexdigest(),
        "domain_counts": dict(audit.domain_counts),
    }


def train_curated_candidate(
    dataset: str | Path,
    baseline_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    report_path: str | Path,
    *,
    steps: int = 100,
    device: str | None = None,
    max_new_tokens: int = 32,
    trainer_factory: Callable[..., Any] = TrainingExperiment,
    evaluator: Any | None = None,
) -> dict[str, Any]:
    """Train a new checkpoint from approved records and record a held-out gate."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    dataset_path = Path(dataset).expanduser().resolve()
    baseline = Path(baseline_checkpoint).expanduser().resolve()
    candidate = Path(candidate_checkpoint).expanduser().resolve()
    report = Path(report_path).expanduser().resolve()
    if not baseline.is_file():
        raise FileNotFoundError(f"baseline checkpoint not found: {baseline}")
    if len({dataset_path, baseline, candidate, report}) != 4:
        raise ValueError("dataset, baseline, candidate, and report paths must be distinct")
    for destination, label in ((candidate, "candidate checkpoint"), (report, "report")):
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {label}: {destination}")
    dataset_info = validate_approved_dataset(dataset_path)

    trainer = trainer_factory(TrainingConfig(steps=steps), device=device)
    training = trainer.run(dataset_path, candidate)
    if not candidate.is_file():
        raise FileNotFoundError("training runner did not produce the candidate checkpoint")

    engine = evaluator or CheckpointEvaluator(max_new_tokens=max_new_tokens)
    baseline_eval = engine.evaluate(baseline, name="tara-heldout-capability-v1")
    candidate_eval = engine.evaluate(candidate, name="tara-heldout-capability-v1")
    decision = engine.compare(baseline_eval, candidate_eval)
    baseline_scores = {item.category: item.score for item in baseline_eval.benchmark.categories}
    candidate_scores = {item.category: item.score for item in candidate_eval.benchmark.categories}
    deltas = [
        {"category": name, "baseline": baseline_scores[name], "candidate": candidate_scores[name],
         "delta": candidate_scores[name] - baseline_scores[name]}
        for name in sorted(baseline_scores.keys() & candidate_scores.keys())
    ]
    payload = {
        "schema_version": 1,
        "dataset": dataset_info,
        "training": training.as_dict() if hasattr(training, "as_dict") else training,
        "baseline": baseline_eval.as_dict(),
        "candidate": candidate_eval.as_dict(),
        "category_deltas": deltas,
        "decision": decision.as_dict(),
        "report_fingerprint": hashlib.sha256(json.dumps({
            "dataset": dataset_info["dataset_fingerprint"],
            "baseline": baseline_eval.fingerprint,
            "candidate": candidate_eval.fingerprint,
            "decision": decision.fingerprint,
        }, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
        "interpretation": "A passing structural audit is not a correctness guarantee. The held-out capability gate is a limited regression check, not proof of general intelligence. No checkpoint is promoted automatically.",
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a TARA candidate from human-approved teacher data")
    parser.add_argument("dataset", help="curated JSONL containing only explicitly approved records")
    parser.add_argument("baseline_checkpoint")
    parser.add_argument("candidate_checkpoint", help="new path; existing checkpoints are never overwritten")
    parser.add_argument("report", help="new JSON evidence report path")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    result = train_curated_candidate(
        args.dataset, args.baseline_checkpoint, args.candidate_checkpoint, args.report,
        steps=args.steps, device=args.device, max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if not result["decision"]["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
