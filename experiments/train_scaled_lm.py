"""Train TARA's PC-friendly scaled language model and report diagnostics.

This is an experiment, not the production agent. It uses a research-based
adaptive optimizer and records objective metrics before generation changes.
"""

from experiments.scaled_lm_diagnostics import (
    CORPUS,
    CONTEXT_LENGTH,
    EMBEDDING_DIM,
    FF_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    VALIDATION_FRACTION,
    VOCAB_SIZE,
    build_experiment,
    evaluate,
)

from src.gradient_clipping import clip_grad_norm_
from src.optimizers import AdamW
from src.schedulers import CosineAnnealing


STEPS = 100
BATCH_SIZE = 4
LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 7


def train(steps=STEPS):
    if steps <= 0:
        raise ValueError("steps must be positive")

    model, tokenizer, train_dataset, validation_dataset = build_experiment(CORPUS)
    scheduler = CosineAnnealing(
        LEARNING_RATE,
        total_steps=steps,
        min_lr=MIN_LEARNING_RATE,
    )
    optimizer = AdamW(
        model.parameters(),
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    initial_train = evaluate(model, train_dataset, BATCH_SIZE)
    initial_validation = evaluate(model, validation_dataset, BATCH_SIZE)
    history = []
    step = 0
    epoch = 0

    while step < steps:
        for batch in train_dataset.iter_batches(
            BATCH_SIZE,
            shuffle=True,
            seed=SEED + epoch,
        ):
            if step >= steps:
                break
            model.zero_grad()
            total_loss = None
            total_tokens = 0
            for inputs, targets in batch:
                loss = model.loss(inputs, targets)
                weighted = loss * len(targets)
                total_loss = weighted if total_loss is None else total_loss + weighted
                total_tokens += len(targets)
            loss = total_loss / total_tokens
            loss.backward()
            gradient_norm = clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            learning_rate = scheduler.get_lr(step)
            optimizer.step(learning_rate=learning_rate)

            history.append({
                "step": step,
                "loss": loss.data,
                "learning_rate": learning_rate,
                "gradient_norm": gradient_norm,
            })
            step += 1
        epoch += 1

    final_train = evaluate(model, train_dataset, BATCH_SIZE)
    final_validation = evaluate(model, validation_dataset, BATCH_SIZE)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "initial_train": initial_train,
        "initial_validation": initial_validation,
        "final_train": final_train,
        "final_validation": final_validation,
        "history": history,
        "optimizer": optimizer.state_dict(),
        "config": {
            "vocab_size": VOCAB_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "ff_dim": FF_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "context_length": CONTEXT_LENGTH,
            "validation_fraction": VALIDATION_FRACTION,
            "steps": steps,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "min_learning_rate": MIN_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
        },
    }


if __name__ == "__main__":
    result = train()
    print("Initial train:", result["initial_train"])
    print("Final train:", result["final_train"])
    print("Initial validation:", result["initial_validation"])
