"""Run TARA's trained model through generation, execution, and improvement.

Example:
    python discover_polyglot.py sorting --checkpoint checkpoints/algorithm_lm.pt
"""

from __future__ import annotations

import argparse

from src.model_runtime import load_checkpoint, generate_text
from src.polyglot import PolyglotBenchmark, SelfImprovementEngine, TARAAlgorithmModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and verify algorithm candidates with TARA")
    parser.add_argument("problem", help="registered problem name, e.g. sorting")
    parser.add_argument("--checkpoint", default="checkpoints/algorithm_lm.pt")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=512)
    args = parser.parse_args()

    runtime = TARAAlgorithmModel(args.checkpoint, max_new_tokens=args.tokens)
    generator = runtime.candidate_generator()
    engine = SelfImprovementEngine(PolyglotBenchmark())
    result = engine.improve(args.problem, generator, rounds=args.rounds, candidates_per_round=args.candidates)

    print(f"rounds={result.rounds} candidates_tested={len(result.history)}")
    if result.best is None:
        print("No parseable candidate was generated.")
        return
    best = result.best
    print(f"best_language={best.candidate.language}")
    print(f"correctness={best.correctness:.3f}")
    print(f"runtime_ms={best.total_runtime_ms:.3f}")
    print(f"verified={best.verified}")
    print("--- source ---")
    print(best.candidate.source)
    if best.failures:
        print("--- failures ---")
        for failure in best.failures:
            print(failure)


if __name__ == "__main__":
    main()
