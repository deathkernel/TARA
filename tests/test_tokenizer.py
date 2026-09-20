from src.tokenizer import CharTokenizer


def test_character_tokenizer_round_trip():
    tokenizer = CharTokenizer("hello tara")
    encoded = tokenizer.encode("tara")
    assert tokenizer.decode(encoded) == "tara"
    assert tokenizer.vocab_size == len(set("hello tara")) + 1


def test_unknown_character_uses_unk_token():
    tokenizer = CharTokenizer("abc")
    assert tokenizer.decode(tokenizer.encode("ax")) == "a<UNK>"
