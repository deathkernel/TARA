"""Generate candidate algorithm ideas from a trained TARA checkpoint.

Verification is intentionally not performed here: generated code/text must be
passed to a separate, restricted verifier before being accepted.
"""

import argparse

from src.model_runtime import generate_text, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("problem")
    parser.add_argument("--checkpoint", default="checkpoints/algorithm_lm.pt")
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if args.candidates <= 0:
        raise ValueError("candidates must be positive")

    model, tokenizer = load_checkpoint(args.checkpoint)
    prompt = "Problem: " + args.problem + "\nApproach:"

    for index in range(1, args.candidates + 1):
        text = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
        )
        print(f"\n=== Candidate {index} ===\n{text}")


if __name__ == "__main__":
    main()
