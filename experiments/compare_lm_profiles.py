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
