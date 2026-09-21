"""Measure how BPE vocabulary size changes token compression.

Research question:
Does increasing the BPE vocabulary reduce sequence length enough to justify
using a larger token vocabulary in TARA's tiny language model?

This experiment intentionally measures tokenization only. It does not train a
language model, so several configurations can be compared quickly before
spending time on full LM training.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.text_dataset import load_tinystories_text
from src.tokenizer import BPETokenizer, CharTokenizer


TEXT_CHARS = 4096
BPE_VOCAB_SIZES = (64, 96, 128)


def main():
    text = load_tinystories_text(max_chars=TEXT_CHARS, split="train")
    char = CharTokenizer(text)
    char_tokens = len(char.encode(text))
    minimum_vocab = len(set(text)) + 1

    print("TARA BPE vocabulary-size sweep")
    print(f"Training text characters: {len(text)}")
    print(f"Character tokens: {char_tokens}")
    print(f"Minimum BPE vocabulary for this corpus: {minimum_vocab}")
    print()
    print("vocab_size | tokens | token/char | compression")
    print("-----------+--------+------------+------------")

    for requested_size in BPE_VOCAB_SIZES:
        vocab_size = max(requested_size, minimum_vocab)
        tokenizer = BPETokenizer(text, vocab_size=vocab_size)
        token_count = len(tokenizer.encode(text))
        ratio = token_count / char_tokens
        compression = (1.0 - ratio) * 100.0
        print(
            f"{tokenizer.vocab_size:10d} | "
            f"{token_count:6d} | "
            f"{ratio:10.4f} | "
            f"{compression:9.2f}%"
        )


if __name__ == "__main__":
    main()
