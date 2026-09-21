from src.tokenizer import BPETokenizer, CharTokenizer


def test_character_tokenizer_round_trip():
    tokenizer = CharTokenizer("hello tara")
    encoded = tokenizer.encode("tara")
    assert tokenizer.decode(encoded) == "tara"
    assert tokenizer.vocab_size == len(set("hello tara")) + 1


def test_unknown_character_uses_unk_token():
    tokenizer = CharTokenizer("abc")
    assert tokenizer.decode(tokenizer.encode("ax")) == "a<UNK>"


def test_bpe_round_trip():
    tokenizer = BPETokenizer("hello hello hello", vocab_size=8)
    encoded = tokenizer.encode("hello")
    assert tokenizer.decode(encoded) == "hello"
    assert tokenizer.vocab_size == 8


def test_bpe_learns_frequent_subword():
    tokenizer = BPETokenizer("abababab", vocab_size=4)
    assert "ab" in tokenizer.itos
    assert tokenizer.encode_tokens("abab") == ["ab", "ab"]


def test_bpe_is_deterministic():
    text = "the cat sat on the mat. the cat sat."
    first = BPETokenizer(text, vocab_size=16)
    second = BPETokenizer(text, vocab_size=16)
    assert first.itos == second.itos
    assert first.merges == second.merges
    assert first.encode(text) == second.encode(text)


def test_bpe_rejects_too_small_vocab():
    try:
        BPETokenizer("abc", vocab_size=3)
        assert False
    except ValueError:
        pass
