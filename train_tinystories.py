"""Train and evaluate TARA on a small streamed TinyStories slice.

This experiment separates train and validation text so we can measure
optimization and generalization without changing the model architecture.
The slices stay deliberately small because TARA uses a scalar autodiff
engine and is intended to run on a normal PC.
"""

from src.language_model import TinyLanguageModel
from src.text_dataset import load_tinystories_text
from src.tokenizer import CharTokenizer


MAX_TRAIN_CHARS = 512
MAX_VALIDATION_CHARS = 512
EMBEDDING_DIM = 3
FF_DIM = 6
STEPS = 80
LEARNING_RATE = 0.03
CONTEXT_LENGTH = 12
SEED = 7


def token_loss(model, ids):
    """Return token-weighted next-token loss for a token sequence."""
    if len(ids) < 2:
        raise ValueError("dataset must contain at least two tokens")

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

    return total_loss / total_tokens


def train(train_corpus=None, steps=STEPS, learning_rate=LEARNING_RATE):
    if train_corpus is None:
        train_corpus = load_tinystories_text(
            max_chars=MAX_TRAIN_CHARS, split="train"
        )

    tokenizer = CharTokenizer(train_corpus)
    model = TinyLanguageModel(
        tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        ff_dim=FF_DIM,
        seed=SEED,
    )
    train_ids = tokenizer.encode(train_corpus)

    if len(train_ids) < 2:
        raise ValueError("train_corpus must contain at least two tokens")

    for step in range(steps):
        model.zero_grad()
        loss = token_loss(model, train_ids)
        loss.backward()
        for parameter in model.parameters():
            parameter.data -= learning_rate * parameter.grad

        if step % 10 == 0 or step == steps - 1:
            print(f"step={step:3d} train_loss={loss.data:.6f}")

    return model, tokenizer, train_corpus


def evaluate(model, tokenizer, validation_corpus):
    """Measure next-token loss on unseen validation text."""
    validation_ids = tokenizer.encode(validation_corpus)
    if len(validation_ids) < 2:
        raise ValueError("validation_corpus must contain at least two tokens")
    model.zero_grad()
    loss = token_loss(model, validation_ids)
    return loss.data


def generate(model, tokenizer, prompt, length=80):
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt must not be empty")
    for _ in range(length):
        context = ids[-CONTEXT_LENGTH:]
        ids.append(model.next_token(context))
    return tokenizer.decode(ids)


if __name__ == "__main__":
    model, tokenizer, train_corpus = train()
    validation_corpus = load_tinystories_text(
        max_chars=MAX_VALIDATION_CHARS, split="validation"
    )
    validation_loss = evaluate(model, tokenizer, validation_corpus)

    print(f"\nTrain characters: {len(train_corpus)}")
    print(f"Validation characters: {len(validation_corpus)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Validation loss: {validation_loss:.6f}")

    print("\nGenerated from unseen validation prompt:")
    print(generate(model, tokenizer, validation_corpus[:20]))
