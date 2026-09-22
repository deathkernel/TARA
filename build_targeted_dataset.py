"""Build a targeted synthetic dataset from a capability diagnosis report."""
from __future__ import annotations

import argparse
import json

from src.targeted_training import build_targeted_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build targeted training data from TARA benchmark failures")
    parser.add_argument("diagnosis", help="failure diagnosis JSON")
    parser.add_argument("--output", default="data/targeted_training.jsonl")
    parser.add_argument("--variants-per-failure", type=int, default=3)
    parser.add_argument("--max-examples", type=int, default=128)
    args = parser.parse_args()

    examples = build_targeted_dataset(
        args.diagnosis,
        args.output,
        variants_per_failure=args.variants_per_failure,
        max_examples=args.max_examples,
    )
    print(json.dumps({
        "output": args.output,
        "examples": len(examples),
        "provenance": "synthetic-from-benchmark-failure",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
