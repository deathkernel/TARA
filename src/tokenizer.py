"""Small tokenizers used by TARA.

Research basis:
- Kudo & Richardson (2018), SentencePiece, motivates subword tokenization
  as a practical language-independent approach.
- OpenAI's tiktoken uses byte-pair encoding (BPE); TARA studies the idea with
  a deliberately small, dependency-free implementation before using an
  external tokenizer.
"""

from collections import Counter


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


class BPETokenizer:
    """Dependency-free educational BPE tokenizer trained from raw text.

    This is intentionally not a drop-in implementation of SentencePiece or
    tiktoken. It starts from characters and repeatedly merges the most
    frequent adjacent token pair until the requested vocabulary size is
    reached. The learned merge order is then replayed during encoding.
    """

    UNK = "<UNK>"

    def __init__(self, text, vocab_size=128):
        if not text:
            raise ValueError("training text must not be empty")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        base_tokens = sorted(set(text))
        if vocab_size < len(base_tokens) + 1:
            raise ValueError("vocab_size must fit UNK plus all training characters")

        self.itos = [self.UNK] + base_tokens
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        self.merges = []
        self._train(text, vocab_size)

    @property
    def vocab_size(self):
        return len(self.itos)

    def _train(self, text, target_vocab_size):
        symbols = list(text)
        while len(self.itos) < target_vocab_size:
            pair_counts = Counter(zip(symbols, symbols[1:]))
            if not pair_counts:
                break

            # Deterministic tie-breaking makes training reproducible.
            best_pair, best_count = min(
                pair_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if best_count < 2:
                break

            merged = best_pair[0] + best_pair[1]
            if merged in self.stoi:
                break

            self.merges.append(best_pair)
            self.itos.append(merged)
            self.stoi[merged] = len(self.itos) - 1
            symbols = self._merge_pair(symbols, best_pair)

    @staticmethod
    def _merge_pair(symbols, pair):
        merged = []
        i = 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return merged

    def _apply_merges(self, symbols):
        for pair in self.merges:
            symbols = self._merge_pair(symbols, pair)
        return symbols

    def encode_tokens(self, text):
        """Return learned token strings, useful for inspecting tokenization."""
        if not text:
            return []
        symbols = list(text)
        symbols = [symbol if symbol in self.stoi else self.UNK for symbol in symbols]
        return self._apply_merges(symbols)

    def encode(self, text):
        """Encode text into deterministic integer token IDs."""
        return [self.stoi.get(token, self.stoi[self.UNK]) for token in self.encode_tokens(text)]

    def decode(self, ids):
        """Decode token IDs back into text."""
        pieces = []
        for index in ids:
            if index < 0 or index >= self.vocab_size:
                pieces.append(self.UNK)
            else:
                pieces.append(self.itos[index])
        return "".join(pieces)
