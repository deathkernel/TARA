"""Evaluate TARA on a held-out text stream from a separate corpus.

The ordinary train/validation split is useful for optimization diagnostics,
but a second, unseen text stream gives a stronger signal against simply
memorizing the training corpus. The tokenizer is learned only from the
training corpus; the held-out text is encoded with that fixed tokenizer.
"""

from experiments.train_scaled_lm import train
from experiments.scaled_lm_diagnostics import evaluate
from src.language_dataset import CausalTextDataset, dataset_statistics


HELDOUT_CORPUS = (
    "tara solves a fresh problem by combining context and evidence. "
    "a useful model predicts patterns that transfer to another sentence. "
    "reasoning can improve when a goal is divided into smaller steps. "
)


def build_heldout_dataset(tokenizer, text=HELDOUT_CORPUS, context_length=64):
    """Encode a separate corpus with the already-trained tokenizer."""
    token_ids = tokenizer.encode(text)
    return CausalTextDataset(token_ids, context_length)


def evaluate_heldout(steps=100):
    """Train the scaled profile, then evaluate it on unseen text."""
    result = train(steps=steps)
    dataset = build_heldout_dataset(
        result["tokenizer"],
        context_length=result["config"]["context_length"],
    )
    metrics = evaluate(
        result["model"],
        dataset,
        result["config"]["batch_size"],
    )
    statistics = dataset_statistics(
        dataset,
        unk_id=result["tokenizer"].stoi[result["tokenizer"].UNK],
    )
    return {"training": result, "heldout": metrics, "statistics": statistics}


if __name__ == "__main__":
    result = evaluate_heldout()
    print("Held-out statistics:", result["statistics"])
    print("Held-out:", result["heldout"])
