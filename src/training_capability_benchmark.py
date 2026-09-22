"""Measure whether a trained TARA checkpoint improved on its training task.

The benchmark compares a freshly initialized model with the saved checkpoint on
an identical deterministic validation corpus. It is intentionally modest: a
lower validation loss demonstrates learning on the supplied corpus, but does
not by itself establish broad reasoning capability or generalization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .training_pipeline import (
    CHECKPOINT_FORMAT_VERSION,
    _fingerprint,
    _tokenize_corpus,
    evaluate,
    load_training_texts,
    split_texts,
)
from .torch_language_model import FastTinyLanguageModel
from .tokenizer import CharTokenizer


@dataclass(frozen=True)
class CapabilityBenchmarkReport:
    checkpoint: str
    dataset: str
    dataset_fingerprint: str
    device: str
    baseline_validation_loss: float | None
    trained_validation_loss: float | None
    loss_delta: float | None
    relative_improvement: float | None
    trained_better: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported TARA training checkpoint format")
    return payload


def benchmark_checkpoint(
    checkpoint: str | Path,
    dataset: str | Path,
    *,
    device: str = "cpu",
) -> CapabilityBenchmarkReport:
    """Compare checkpoint validation loss with the matching fresh model."""
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")

    checkpoint_path = Path(checkpoint)
    dataset_path = Path(dataset)
    payload = _load_checkpoint(checkpoint_path, target_device)
    texts = load_training_texts(dataset_path)
    dataset_fingerprint = _fingerprint(texts)
    if payload.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("dataset fingerprint differs from checkpoint")

    config = payload["config"]
    validation_split = float(config["validation_split"])
    seed = int(config["seed"])
    context = int(config["context"])
    batch_size = int(config["batch_size"])
    train_texts, validation_texts = split_texts(texts, validation_split, seed)
    tokenizer = CharTokenizer("\n".join(train_texts))
    validation_tokens = _tokenize_corpus(validation_texts, tokenizer, context) if validation_texts else None

    model_config = payload["model_config"]
    baseline = FastTinyLanguageModel(**model_config, seed=seed).to(target_device)
    baseline_loss = evaluate(baseline, validation_tokens, batch_size, context, target_device)

    trained = FastTinyLanguageModel(**model_config, seed=seed).to(target_device)
    trained.load_state_dict(payload["model_state"])
    trained_loss = evaluate(trained, validation_tokens, batch_size, context, target_device)

    if baseline_loss is None or trained_loss is None:
        delta = None
        improvement = None
        better = False
    else:
        delta = trained_loss - baseline_loss
        improvement = (baseline_loss - trained_loss) / baseline_loss if baseline_loss else None
        better = trained_loss < baseline_loss

    return CapabilityBenchmarkReport(
        checkpoint=str(checkpoint_path),
        dataset=str(dataset_path),
        dataset_fingerprint=dataset_fingerprint,
        device=str(target_device),
        baseline_validation_loss=baseline_loss,
        trained_validation_loss=trained_loss,
        loss_delta=delta,
        relative_improvement=improvement,
        trained_better=better,
    )
