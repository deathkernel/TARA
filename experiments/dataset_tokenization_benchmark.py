"""Compare tokenization statistics across TARA datasets.

This experiment measures corpus length and token compression only. It does not
claim that one dataset or tokenizer is better for language modeling.
"""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.text_dataset import list_datasets, load_dataset_text
from src.tokenizer import BPETokenizer, CharTokenizer


MAX_CHARS = 4096
BPE_VOCAB_SIZE = 128


def main():
    print("TARA cross-dataset tokenization benchmark")
    print(f"Character budget: {MAX_CHARS}")
    print(f"BPE vocabulary target: {BPE_VOCAB_SIZE}")
    print()

    for dataset_name in list_datasets():
        started = time.perf_counter()
        train = load_dataset_text(dataset_name, MAX_CHARS, "train")
        validation = load_dataset_text(
            dataset_name, MAX_CHARS // 4, "validation"
        )

        char = CharTokenizer(train)
        minimum_vocab = len(set(train)) + 1
        bpe_vocab_size = max(BPE_VOCAB_SIZE, minimum_vocab)
        bpe = BPETokenizer(train, vocab_size=bpe_vocab_size)

        train_chars = len(train)
        char_tokens = len(char.encode(train))
        bpe_tokens = len(bpe.encode(train))
        validation_chars = len(validation)
        validation_bpe_tokens = len(bpe.encode(validation))
        elapsed = time.perf_counter() - started

        print(f"[{dataset_name}]")
        print(f"  train chars: {train_chars}")
        print(f"  validation chars: {validation_chars}")
        print(f"  character tokens: {char_tokens}")
        print(f"  BPE tokens: {bpe_tokens}")
        print(f"  BPE vocabulary: {bpe.vocab_size}")
        print(f"  BPE token/char: {bpe_tokens / train_chars:.4f}")
        print(f"  validation BPE tokens: {validation_bpe_tokens}")
        print(f"  setup time: {elapsed:.3f}s")
        print()


if __name__ == "__main__":
    main()
