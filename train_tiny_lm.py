"""Train and validate TARA on a tiny local corpus.

Research basis:
- GPT-style autoregressive language modeling predicts the next token.
- Mini-batch gradient updates process a small group of examples per update.
- Validation loss is measured on held-out tokens that are never used for
  parameter updates.

The experiment intentionally remains small enough for a normal PC.
"""

from src.gradient_clipping import clip_grad_norm_
from src.language_dataset import build_causal_datasets
from src.language_model import TinyLanguageModel
from src.schedulers import CosineAnnealing
from src.tokenizer import CharTokenizer


CORPUS = "tara learns. tara reasons. tara learns. tara reasons. "
EMBEDDING_DIM = 3
FF_DIM = 6
STEPS = 80
LEARNING_RATE = 0.03
MIN_LEARNING_RATE = 0.003
MAX_GRAD_NORM = 1.0
CONTEXT_LENGTH = 12
BATCH_SIZE = 2
VALIDATION_FRACTION = 0.2
SEED = 7


def _mean_loss(model, dataset, batch_size):
    """Evaluate mean next-token loss without updating model parameters."""
    total_loss = 0.0
    total_tokens = 0

    for batch in dataset.iter_batches(batch_size, shuffle=False):
        for inputs, targets in batch:
            loss = model.loss(inputs, targets)
            total_loss += loss.data * len(targets)
            total_tokens += len(targets)

    if total_tokens == 0:
        raise ValueError("dataset must contain at least one target token")
    return total_loss / total_tokens


def _batch_loss(model, batch):
    """Return a token-weighted mean loss for one mini-batch."""
    total_loss = None
    total_tokens = 0
    for inputs, targets in batch:
        loss = model.loss(inputs, targets)
        weighted_loss = loss * len(targets)
        total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        total_tokens += len(targets)

    if total_tokens == 0:
        raise ValueError("batch must contain at least one target token")
    return total_loss / total_tokens


def train(corpus=CORPUS, steps=STEPS, learning_rate=LEARNING_RATE):
    tokenizer = CharTokenizer(corpus)
    train_dataset, validation_dataset = build_causal_datasets(
        tokenizer,
        corpus,
        context_length=CONTEXT_LENGTH,
        validation_fraction=VALIDATION_FRACTION,
    )
    model = TinyLanguageModel(
        tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        ff_dim=FF_DIM,
        seed=SEED,
    )

    train_batches = list(
        train_dataset.iter_batches(BATCH_SIZE, shuffle=False)
    )
    if not train_batches:
        raise ValueError("training dataset must contain at least one batch")

    scheduler = CosineAnnealing(
        learning_rate,
        total_steps=steps,
        min_lr=min(learning_rate, MIN_LEARNING_RATE),
    )

    for step in range(steps):
        shuffled_batches = list(
            train_dataset.iter_batches(
                BATCH_SIZE,
                shuffle=True,
                seed=SEED + step,
            )
        )
        batch = shuffled_batches[step % len(shuffled_batches)]

        model.zero_grad()
        loss = _batch_loss(model, batch)
        loss.backward()

        current_lr = scheduler.get_lr(step)
        gradient_norm = clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        for parameter in model.parameters():
            parameter.data -= current_lr * parameter.grad

        if step % 10 == 0 or step == steps - 1:
            validation_loss = _mean_loss(model, validation_dataset, BATCH_SIZE)
            print(
                f"step={step:3d} lr={current_lr:.6f} "
                f"grad_norm={gradient_norm:.6f} "
                f"train_loss={loss.data:.6f} "
                f"val_loss={validation_loss:.6f}"
            )

    return model, tokenizer, train_dataset, validation_dataset


def generate(model, tokenizer, prompt, length=40):
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt must not be empty")
    for _ in range(length):
        context = ids[-CONTEXT_LENGTH:]
        ids.append(model.next_token(context))
    return tokenizer.decode(ids)


if __name__ == "__main__":
    model, tokenizer, _, _ = train()
    print("\nGenerated:")
    print(generate(model, tokenizer, "tara "))
