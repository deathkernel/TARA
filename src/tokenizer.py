"""A tiny character-level tokenizer for TARA.

Research basis:
- SentencePiece (Kudo & Richardson, 2018) motivates subword tokenization
  as a practical language-independent approach.
- TARA intentionally starts simpler: characters are deterministic, local,
  and require no external tokenizer dependency.
"""


class CharTokenizer:
    """Map characters to integer token IDs and back."""

    def __init__(self, text):
        if not text:
            raise ValueError("training text must not be empty")
        vocabulary = sorted(set(text))
        self.itos = ["<UNK>"] + vocabulary
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi.get(character, self.stoi["<UNK>"]) for character in text]

    def decode(self, ids):
        return "".join(self.itos[index] if 0 <= index < self.vocab_size else "<UNK>" for index in ids)
