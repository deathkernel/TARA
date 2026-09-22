"""Evaluate and diagnose a real TARA checkpoint with the deterministic suite."""
from __future__ import annotations

import argparse
import json

from src.capability_failure import CapabilityFailureAnalyzer, write_diagnosis
from src.capability_suite import capability_cases
from src.evaluation_orchestrator import CheckpointEvaluator, EvaluationOrchestrator, write_cycle
from src.model_capability_runner import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TARA checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default=None, help="write baseline/candidate evaluation cycle JSON")
    parser.add_argument("--diagnosis", default=None, help="write observed failures and training targets JSON")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    cycle = EvaluationOrchestrator(
        CheckpointEvaluator(max_new_tokens=args.max_new_tokens)
    ).run(args.checkpoint)
    if args.output:
        write_cycle(cycle, args.output)

    evaluation = evaluate_checkpoint(
        args.checkpoint,
        capability_cases(),
        name="tara-capability-v1",
        max_new_tokens=args.max_new_tokens,
    )
    diagnosis = CapabilityFailureAnalyzer().diagnose(evaluation, capability_cases())
    if args.diagnosis:
        write_diagnosis(diagnosis, args.diagnosis)

    payload = cycle.as_dict()
    payload["diagnosis"] = diagnosis.as_dict()
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
