"""Train TARA on a tiny local corpus and generate text.

The experiment uses causal next-token prediction:
inputs = token_ids[:-1]
targets = token_ids[1:]
"""

from src.language_model import TinyLanguageModel
from src.tokenizer import CharTokenizer


CORPUS = "tara learns. tara reasons. tara learns. tara reasons. "
EMBEDDING_DIM = 3
FF_DIM = 6
STEPS = 80
LEARNING_RATE = 0.03
CONTEXT_LENGTH = 12
SEED = 7


def train(corpus=CORPUS, steps=STEPS, learning_rate=LEARNING_RATE):
    tokenizer = CharTokenizer(corpus)
    model = TinyLanguageModel(
        tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        ff_dim=FF_DIM,
        seed=SEED,
    )
    ids = tokenizer.encode(corpus)
    if len(ids) < 2:
        raise ValueError("corpus must contain at least two tokens")

    for step in range(steps):
        model.zero_grad()
        total_loss = None
        total_tokens = 0
        for start in range(0, len(ids) - 1, CONTEXT_LENGTH):
            window = ids[start:start + CONTEXT_LENGTH + 1]
            if len(window) < 2:
                continue
            inputs = window[:-1]
            targets = window[1:]
            loss = model.loss(inputs, targets)
            weighted_loss = loss * len(targets)
            total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
            total_tokens += len(targets)

        if total_tokens == 0:
            raise ValueError("corpus must contain at least one target token")
        total_loss = total_loss / total_tokens
        total_loss.backward()
        for parameter in model.parameters():
            parameter.data -= learning_rate * parameter.grad

        if step % 10 == 0 or step == steps - 1:
            print(f"step={step:3d} loss={total_loss.data:.6f}")

    return model, tokenizer


def generate(model, tokenizer, prompt, length=40):
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt must not be empty")
    for _ in range(length):
        context = ids[-CONTEXT_LENGTH:]
        ids.append(model.next_token(context))
    return tokenizer.decode(ids)


if __name__ == "__main__":
    model, tokenizer = train()
    print("\nGenerated:")
    print(generate(model, tokenizer, "tara "))
