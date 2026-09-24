"""Fast Rust-backed BPE tokenizer for TARA vNext.

Uses Hugging Face tokenizers for efficient BPE training/encoding.
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


class FastBPETokenizer:
    """Train and use a byte-level BPE tokenizer."""

    def __init__(self, text, vocab_size=4096):
        if not isinstance(text, str) or not text:
            raise ValueError("training text must not be empty")
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
            raise TypeError("vocab_size must be an integer")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        self._tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self._tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<UNK>"],
            show_progress=False,
        )
        self._tokenizer.train_from_iterator([text], trainer=trainer)

    @property
    def vocab_size(self):
        return self._tokenizer.get_vocab_size()

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self._tokenizer.encode(text).ids

    def decode(self, ids):
        return self._tokenizer.decode(list(ids))

    def to_json(self):
        return self._tokenizer.to_str()
