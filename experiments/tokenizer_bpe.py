"""Compare character and educational BPE tokenization on TinyStories text.

This experiment changes no model architecture. It measures only sequence
compression and token pieces so we can decide whether BPE is worth integrating
into TARA's language-model training path.
"""

from src.text_dataset import load_tinystories_text
from src.tokenizer import BPETokenizer, CharTokenizer


TEXT_CHARS = 4096
BPE_VOCAB_SIZE = 64


def main():
    text = load_tinystories_text(max_chars=TEXT_CHARS, split="train")
    char = CharTokenizer(text)
    bpe = BPETokenizer(text, vocab_size=BPE_VOCAB_SIZE)

    char_ids = char.encode(text)
    bpe_ids = bpe.encode(text)
    ratio = len(bpe_ids) / len(char_ids)

    print(f"Text characters: {len(text)}")
    print(f"Character tokens: {len(char_ids)}")
    print(f"BPE vocabulary: {bpe.vocab_size}")
    print(f"BPE tokens: {len(bpe_ids)}")
    print(f"BPE token/character ratio: {ratio:.4f}")
    print(f"Compression: {(1.0 - ratio) * 100:.2f}%")

    print("\nFirst learned merges:")
    for left, right in bpe.merges[:20]:
        print(repr(left + right))

    prompt = text[:80]
    print("\nCharacter pieces:")
    print(char.encode(prompt))
    print("\nBPE pieces:")
    print(bpe.encode_tokens(prompt))

    assert char.decode(char_ids) == text
    assert bpe.decode(bpe_ids) == text


if __name__ == "__main__":
    main()
