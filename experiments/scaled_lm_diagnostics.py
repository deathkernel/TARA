"""Measure whether TARA's scaled language model is learning useful structure.

This experiment is deliberately diagnostic rather than a decoding hack. It
tracks objective metrics and corpus statistics so scaling and data quality can
be evaluated scientifically before changing generation.
"""

import math

from src.language_dataset import build_causal_datasets, dataset_statistics
from src.language_model import TinyLanguageModel
from src.numeric_inference import forward_numeric
from src.tokenizer import BPETokenizer


CORPUS = (
    "tara is a small reasoning architecture. "
    "tara learns language patterns from examples. "
    "tara uses attention to represent context. "
    "tara checks predictions against evidence. "
    "tara stores useful information in memory. "
    "tara can decompose a difficult goal into smaller steps. "
    "a model should learn patterns from varied examples, not one repeated phrase. "
    "good training data can contain questions, answers, descriptions, and short explanations. "
    "context helps a prediction depend on earlier tokens. "
    "evidence helps a reasoning system compare an answer with what it observed. "
    "a useful memory keeps important information and can forget stale details. "
    "a planner can divide a difficult task into smaller goals and verify each result. "
)
VOCAB_SIZE = 64
EMBEDDING_DIM = 32
FF_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
CONTEXT_LENGTH = 64
VALIDATION_FRACTION = 0.2


def _softmax(logits):
    if not logits or any(not math.isfinite(value) for value in logits):
        raise ValueError("logits must be finite and non-empty")
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("invalid probability distribution")
    return [weight / total for weight in weights]


def evaluate(model, dataset, batch_size=4):
    """Return loss, next-token accuracy, and mean predictive entropy."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    total_loss = 0.0
    correct = 0
    token_count = 0
    entropy_sum = 0.0

    for batch in dataset.iter_batches(batch_size, shuffle=False):
        for inputs, targets in batch:
            logits = forward_numeric(model, inputs)
            for row, target in zip(logits, targets):
                probabilities = _softmax(row)
                prediction = max(range(len(row)), key=row.__getitem__)
                if not 0 <= target < len(row):
                    raise IndexError("target token is outside model vocabulary")
                target_probability = max(probabilities[target], 1e-30)
                total_loss += -math.log(target_probability)
                correct += int(prediction == target)
                entropy_sum -= sum(
                    probability * math.log(max(probability, 1e-30))
                    for probability in probabilities
                )
                token_count += 1

    if token_count == 0:
        raise ValueError("dataset must contain at least one target token")

    return {
        "loss": total_loss / token_count,
        "accuracy": correct / token_count,
        "entropy": entropy_sum / token_count,
        "tokens": token_count,
    }


def build_experiment(corpus=CORPUS):
    """Construct the PC-friendly scaled model and BPE causal datasets."""
    tokenizer = BPETokenizer(corpus, vocab_size=VOCAB_SIZE)
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
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        seed=7,
    )
    return model, tokenizer, train_dataset, validation_dataset


def corpus_statistics(tokenizer, train_dataset, validation_dataset):
    """Return token/window statistics for the train/validation streams."""
    unk_id = tokenizer.stoi[tokenizer.UNK]
    return {
        "train": dataset_statistics(train_dataset, unk_id=unk_id),
        "validation": dataset_statistics(validation_dataset, unk_id=unk_id),
    }


if __name__ == "__main__":
    model, tokenizer, train_dataset, validation_dataset = build_experiment()
    print("vocab_size:", tokenizer.vocab_size)
    print("statistics:", corpus_statistics(tokenizer, train_dataset, validation_dataset))
    print("train:", evaluate(model, train_dataset))
    print("validation:", evaluate(model, validation_dataset))
