"""Train TARA on a small streamed slice of TinyStories.

This is the first dataset-backed language-model experiment. The slice is
kept deliberately small because TARA uses a scalar autodiff engine and is
intended to run on a normal PC.
"""

from src.language_model import TinyLanguageModel
from src.text_dataset import load_tinystories_text
from src.tokenizer import CharTokenizer


MAX_CHARS = 512
EMBEDDING_DIM = 3
FF_DIM = 6
STEPS = 80
LEARNING_RATE = 0.03
CONTEXT_LENGTH = 12
SEED = 7


def train(corpus=None, steps=STEPS, learning_rate=LEARNING_RATE):
    if corpus is None:
        corpus = load_tinystories_text(max_chars=MAX_CHARS, split="train")
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

        total_loss = total_loss / total_tokens
        total_loss.backward()
        for parameter in model.parameters():
            parameter.data -= learning_rate * parameter.grad

        if step % 10 == 0 or step == steps - 1:
            print(f"step={step:3d} loss={total_loss.data:.6f}")

    return model, tokenizer, corpus


def generate(model, tokenizer, prompt, length=80):
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt must not be empty")
    for _ in range(length):
        context = ids[-CONTEXT_LENGTH:]
        ids.append(model.next_token(context))
    return tokenizer.decode(ids)


if __name__ == "__main__":
    model, tokenizer, corpus = train()
    print(f"\nDataset characters: {len(corpus)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print("\nGenerated:")
    print(generate(model, tokenizer, corpus[:20]))
