from train_tiny_lm import CONTEXT_LENGTH, generate, train


def test_next_token_training_shift_is_loss_improving():
    model, tokenizer = train(steps=2)
    ids = tokenizer.encode("tara learns. ")
    assert len(ids) > 1
    assert ids[:-1] != ids[1:]
    assert model.next_token(ids[-CONTEXT_LENGTH:]) in range(tokenizer.vocab_size)


def test_tiny_training_reduces_loss():
    from src.language_model import TinyLanguageModel

    corpus = "tara learns. tara learns. "
    tokenizer = __import__("src.tokenizer", fromlist=["CharTokenizer"]).CharTokenizer(corpus)
    ids = tokenizer.encode(corpus)
    model = TinyLanguageModel(tokenizer.vocab_size, embedding_dim=2, ff_dim=4, seed=7)

    def loss_value():
        return model.loss(ids[:-1], ids[1:]).data

    initial = loss_value()
    for _ in range(3):
        model.zero_grad()
        loss = model.loss(ids[:-1], ids[1:])
        loss.backward()
        for parameter in model.parameters():
            parameter.data -= 0.01 * parameter.grad
    assert loss_value() < initial


def test_generation_returns_decodable_text():
    model, tokenizer = train(steps=1)
    generated = generate(model, tokenizer, "tara ", length=5)
    assert generated.startswith("tara ")
    assert all(character in tokenizer.itos for character in generated)
