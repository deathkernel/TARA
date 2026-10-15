"""Evaluate baseline and candidate on the fixed held-out capability suite.

The report is evidence-only and never promotes or replaces a checkpoint.
"""
from __future__ import annotations
import argparse, json
from hashlib import sha256
from pathlib import Path
from typing import Any
from src.evaluation_orchestrator import CheckpointEvaluator

def evaluate_heldout(baseline_checkpoint: str | Path, candidate_checkpoint: str | Path,
                     report_path: str | Path, *, max_new_tokens: int = 32,
                     overall_tolerance: float = 0.0, category_tolerance: float = 0.0,
                     evaluator: CheckpointEvaluator | None = None) -> dict[str, Any]:
    baseline_path = Path(baseline_checkpoint).expanduser().resolve()
    candidate_path = Path(candidate_checkpoint).expanduser().resolve()
    destination = Path(report_path).expanduser().resolve()
    if not baseline_path.is_file():
        raise FileNotFoundError(f"baseline checkpoint not found: {baseline_path}")
    if not candidate_path.is_file():
        raise FileNotFoundError(f"candidate checkpoint not found: {candidate_path}")
    if baseline_path == candidate_path:
        raise ValueError("baseline and candidate must be different checkpoints")
    if destination in {baseline_path, candidate_path}:
        raise ValueError("report path must not overwrite a checkpoint")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation report: {destination}")
    engine = evaluator or CheckpointEvaluator(max_new_tokens=max_new_tokens,
        overall_tolerance=overall_tolerance, category_tolerance=category_tolerance)
    baseline = engine.evaluate(baseline_path, name="tara-heldout-capability-v1")
    candidate = engine.evaluate(candidate_path, name="tara-heldout-capability-v1")
    decision = engine.compare(baseline, candidate)
    base_categories = {item.category: item.score for item in baseline.benchmark.categories}
    candidate_categories = {item.category: item.score for item in candidate.benchmark.categories}
    deltas = [{"category": name, "baseline": base_categories[name],
        "candidate": candidate_categories[name], "delta": candidate_categories[name] - base_categories[name]}
        for name in sorted(base_categories.keys() & candidate_categories.keys())]
    payload = {"schema_version": 1, "benchmark": baseline.benchmark.name,
        "benchmark_fingerprint": baseline.benchmark.fingerprint,
        "baseline": baseline.as_dict(), "candidate": candidate.as_dict(),
        "category_deltas": deltas, "decision": decision.as_dict(),
        "report_fingerprint": sha256(json.dumps({"baseline": baseline.fingerprint,
            "candidate": candidate.fingerprint, "decision": decision.fingerprint},
            sort_keys=True).encode()).hexdigest(),
        "interpretation": "This fixed suite is a regression check, not proof of general intelligence or a replacement for domain-specific held-out tests."}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return payload

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and candidate on TARA's held-out capability suite")
    parser.add_argument("baseline_checkpoint"); parser.add_argument("candidate_checkpoint"); parser.add_argument("report")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--overall-tolerance", type=float, default=0.0)
    parser.add_argument("--category-tolerance", type=float, default=0.0)
    args = parser.parse_args()
    result = evaluate_heldout(args.baseline_checkpoint, args.candidate_checkpoint, args.report,
        max_new_tokens=args.max_new_tokens, overall_tolerance=args.overall_tolerance,
        category_tolerance=args.category_tolerance)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if not result["decision"]["accepted"]:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
