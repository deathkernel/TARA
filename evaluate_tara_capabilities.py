"""Evaluate a real TARA checkpoint with the deterministic capability suite."""
from __future__ import annotations

import argparse
import json

from src.evaluation_orchestrator import CheckpointEvaluator, EvaluationOrchestrator, write_cycle


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TARA checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    cycle = EvaluationOrchestrator(
        CheckpointEvaluator(max_new_tokens=args.max_new_tokens)
    ).run(args.checkpoint)
    if args.output:
        write_cycle(cycle, args.output)
    print(json.dumps(cycle.as_dict(), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
