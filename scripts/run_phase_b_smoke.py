"""Run a tiny, real Phase B train/evaluate/gate cycle on CPU.

This is a pipeline smoke test, not evidence of meaningful intelligence gains.
It writes only to a caller-selected, previously unused output directory.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from src.evaluation_orchestrator import CheckpointEvaluator
from src.learning_experiment import LearningExperiment
from src.training_experiment import TrainingExperiment
from src.training_pipeline import TrainingConfig


def _write_fixture(path: Path) -> None:
    rows = [
        {"problem": f"Add {i} and {i + 1}", "solution": f"return {2 * i + 1}", "tests": f"assert solve({i}) == {2 * i + 1}"}
        for i in range(1, 17)
    ]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run_phase_b_smoke(
    output_dir: str | Path,
    *,
    steps: int = 2,
    max_new_tokens: int = 8,
    require_improvement: bool = False,
) -> dict[str, Any]:
    """Train baseline + resumed candidate, evaluate both, and persist evidence."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    expected = (
        "phase_b_dataset.jsonl",
        "baseline.pt",
        "baseline.training.json",
        "baseline.metrics.jsonl",
        "candidate.pt",
        "candidate.training.json",
        "candidate.metrics.jsonl",
        "phase_b_report.json",
        "learning_experiment.json",
    )
    existing = [name for name in expected if (root / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite Phase B artifacts: " + ", ".join(existing)
        )

    dataset = root / "phase_b_dataset.jsonl"
    baseline_checkpoint = root / "baseline.pt"
    candidate_checkpoint = root / "candidate.pt"
    _write_fixture(dataset)

    config = TrainingConfig(
        steps=steps,
        batch_size=2,
        context=16,
        embedding_dim=8,
        ff_dim=16,
        heads=2,
        num_layers=1,
        dropout=0.0,
        lr=1e-3,
        validation_split=0.25,
        seed=2718,
        log_every=1,
        early_stopping_patience=max(steps + 2, 3),
    )
    baseline_training = TrainingExperiment(config=config, device="cpu").run(
        dataset,
        baseline_checkpoint,
        metrics_path=root / "baseline.metrics.jsonl",
        manifest_path=root / "baseline.training.json",
    )

    # Resume from the baseline into a distinct candidate path. The baseline is
    # never overwritten, and the resume contract checks the dataset fingerprint.
    candidate_config = TrainingConfig(**{**config.__dict__, "steps": steps})
    candidate_training = TrainingExperiment(config=candidate_config, device="cpu").run(
        dataset,
        candidate_checkpoint,
        resume=baseline_checkpoint,
        metrics_path=root / "candidate.metrics.jsonl",
        manifest_path=root / "candidate.training.json",
    )

    evaluator = CheckpointEvaluator(max_new_tokens=max_new_tokens)
    learning = LearningExperiment(evaluator=evaluator).run(
        baseline_checkpoint,
        train_candidate=lambda _baseline: candidate_checkpoint,
        output=root / "learning_experiment.json",
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "phase-b-pipeline-smoke",
        "interpretation": (
            "Pipeline execution evidence only; benchmark scores are not a claim "
            "of general intelligence or a guaranteed improvement."
        ),
        "output_dir": str(root),
        "dataset": str(dataset),
        "dataset_fingerprint": baseline_training.dataset_fingerprint,
        "baseline_training": baseline_training.as_dict(),
        "candidate_training": candidate_training.as_dict(),
        "learning_experiment": learning.as_dict(),
        "artifacts": {
            "dataset": dataset.name,
            "baseline_checkpoint": baseline_checkpoint.name,
            "candidate_checkpoint": candidate_checkpoint.name,
            "baseline_manifest": "baseline.training.json",
            "candidate_manifest": "candidate.training.json",
            "learning_report": "learning_experiment.json",
        },
        "smoke_passed": (
            baseline_checkpoint.is_file()
            and candidate_checkpoint.is_file()
            and baseline_training.final_step > baseline_training.start_step
            and candidate_training.final_step > candidate_training.start_step
            and learning.baseline is not None
            and learning.candidate is not None
        ),
        "promotion_accepted": learning.decision.accepted,
    }
    report_path = root / "phase_b_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    if not report["smoke_passed"]:
        raise RuntimeError(f"Phase B smoke run did not complete; see {report_path}")
    if require_improvement and not learning.decision.accepted:
        raise SystemExit(
            "Candidate did not pass the strict improvement/regression gate; "
            f"no promotion performed. Evidence: {report_path}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small real CPU training + checkpoint evaluation + regression-gate smoke test"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="new/empty artifact directory; defaults to a fresh temporary directory",
    )
    parser.add_argument("--steps", type=int, default=2, help="training steps per run (default: 2)")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--require-improvement",
        action="store_true",
        help="exit non-zero unless the candidate strictly improves without regression",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or tempfile.mkdtemp(prefix="tara-phase-b-")
    report = run_phase_b_smoke(
        output_dir,
        steps=args.steps,
        max_new_tokens=args.max_new_tokens,
        require_improvement=args.require_improvement,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
