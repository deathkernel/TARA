"""Compare TARA's tiny and scaled language-model profiles.

The comparison is intentionally measurement-first: both profiles are evaluated
with the same corpus pipeline and report capacity, loss, accuracy, and entropy.
It does not alter generation behavior or add heuristic repetition penalties.
"""

import math

from experiments.scaled_lm_diagnostics import CORPUS, evaluate
from src.language_dataset import build_causal_datasets
from src.language_model import TinyLanguageModel
from src.tokenizer import BPETokenizer


TINY = {"embedding_dim": 8, "ff_dim": 16, "num_heads": 2, "num_layers": 1}
SCALED = {"embedding_dim": 32, "ff_dim": 64, "num_heads": 4, "num_layers": 2}
VOCAB_SIZE = 64
CONTEXT_LENGTH = 64
VALIDATION_FRACTION = 0.2


def parameter_count(model):
    return sum(1 for _ in model.parameters())


def profile_parameter_count(config, vocab_size=VOCAB_SIZE):
    """Return the exact scalar parameter count without building an autodiff graph."""
    d = int(config["embedding_dim"])
    f = int(config["ff_dim"])
    layers = int(config["num_layers"])
    if d <= 0 or f <= 0 or layers <= 0 or vocab_size <= 0:
        raise ValueError("model dimensions and vocab_size must be positive")

    embedding = vocab_size * d
    # LayerNorm(2d) + attention(4d^2 + 4d) + LayerNorm(2d)
    # + FFN(2df + f + d).
    block = 4 * d * d + 2 * d * f + f + 9 * d
    lm_head = d * vocab_size + vocab_size
    return embedding + layers * block + lm_head


def build_profile(config, corpus=CORPUS):
    tokenizer = BPETokenizer(corpus, vocab_size=VOCAB_SIZE)
    train, validation = build_causal_datasets(
        tokenizer,
        corpus,
        context_length=CONTEXT_LENGTH,
        validation_fraction=VALIDATION_FRACTION,
    )
    model = TinyLanguageModel(tokenizer.vocab_size, seed=7, **config)
    return model, train, validation


def compare(corpus=CORPUS):
    results = {}
    for name, config in (("tiny", TINY), ("scaled", SCALED)):
        model, train, validation = build_profile(config, corpus)
        results[name] = {
            "parameters": parameter_count(model),
            "train": evaluate(model, train),
            "validation": evaluate(model, validation),
        }
    return results


if __name__ == "__main__":
    report = compare()
    for name, result in report.items():
        print(name, result)
        assert math.isfinite(result["train"]["loss"])
        assert math.isfinite(result["validation"]["loss"])
