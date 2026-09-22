"""Evaluate a real TARA checkpoint with the deterministic capability suite."""
from __future__ import annotations

import argparse
import json

from src.capability_suite import capability_cases
from src.model_capability_runner import evaluate_checkpoint, write_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TARA checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    evaluation = evaluate_checkpoint(
        args.checkpoint,
        tuple(capability_cases()),
        max_new_tokens=args.max_new_tokens,
    )
    if args.output:
        write_evaluation(evaluation, args.output)
    print(json.dumps(evaluation.as_dict(), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
