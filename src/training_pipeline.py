"""End-to-end PyTorch training pipeline for TARA's learned language core.

Phase 14 connects prepared dataset material to a resumable model checkpoint:

    dataset -> split -> tokenizer -> model -> train/validate -> checkpoint

The checkpoint is intentionally self-contained. It stores model configuration,
tokenizer vocabulary, optimizer state, training progress, metrics, and dataset
fingerprint so a resume can reject incompatible model/tokenizer/dataset state.
Actual training remains an explicit offline operation; importing this module
never starts training.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

from src.tokenizer import CharTokenizer
from src.torch_language_model import FastTinyLanguageModel


CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class TrainingConfig:
    """Validated configuration for a TARA LM training run."""

    steps: int = 1000
    batch_size: int = 32
    context: int = 128
    embedding_dim: int = 64
    ff_dim: int = 128
    heads: int = 4
    lr: float = 3e-4
    validation_split: float = 0.1
    seed: int = 42
    log_every: int = 100
    checkpoint_every: int = 0
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0 or self.context <= 0:
            raise ValueError("steps, batch_size, and context must be positive")
        if self.embedding_dim <= 0 or self.ff_dim <= 0 or self.heads <= 0:
            raise ValueError("model dimensions must be positive")
        if self.embedding_dim % self.heads != 0:
            raise ValueError("embedding_dim must be divisible by heads")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError("validation_split must be in [0, 1)")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.checkpoint_every < 0 or self.grad_clip <= 0:
            raise ValueError("checkpoint_every must be non-negative and grad_clip positive")

    def model_config(self, vocab_size: int) -> dict[str, int]:
        return {
            "vocab_size": vocab_size,
            "embedding_dim": self.embedding_dim,
            "ff_dim": self.ff_dim,
            "num_heads": self.heads,
            "max_context": self.context,
        }


@dataclass(frozen=True)
class TrainingSummary:
    """Final state returned by :meth:`TrainingPipeline.train`."""

    checkpoint: str
    start_step: int
    final_step: int
    train_loss: float
    validation_loss: float | None
    device: str
    dataset_fingerprint: str


def _fingerprint(texts: Iterable[str]) -> str:
    payload = "\n".join(texts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_training_texts(path: str | Path) -> list[str]:
    """Load JSONL/JSON/text records into deterministic training strings.

    Algorithm records use the existing Problem/Approach/Tests format. Generic
    prepared datasets may use ``text``, ``content`` or ``prompt``.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    raw = source.read_text(encoding="utf-8")
    records: list[str] = []

    if source.suffix.lower() in {".txt", ".text"}:
        records = [line.strip() for line in raw.splitlines() if line.strip()]
    else:
        lines = [line for line in raw.splitlines() if line.strip()]
        try:
            payload = json.loads(raw)
            items = payload.get("records", payload) if isinstance(payload, dict) else payload
            if isinstance(items, list):
                iterable = items
            else:
                iterable = [items]
        except json.JSONDecodeError:
            iterable = [json.loads(line) for line in lines]

        for item in iterable:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                if "problem" in item and "solution" in item:
                    text = (
                        "Problem: " + str(item["problem"])
                        + "\nApproach: " + str(item["solution"])
                        + "\nTests: " + str(item.get("tests", ""))
                    )
                else:
                    value = item.get("text", item.get("content", item.get("prompt")))
                    if value is None:
                        raise ValueError("dataset record has no text/content/prompt field")
                    text = str(value).strip()
            else:
                raise ValueError("dataset record must be a string or object")
            if text:
                records.append(text)

    if not records:
        raise ValueError("training dataset is empty")
    return records


def split_texts(texts: list[str], validation_split: float, seed: int) -> tuple[list[str], list[str]]:
    """Deterministically split examples without leaking order from the source."""
    if not texts:
        raise ValueError("texts must not be empty")
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be in [0, 1)")
    indices = list(range(len(texts)))
    random.Random(seed).shuffle(indices)
    if validation_split == 0 or len(texts) < 2:
        return [texts[i] for i in indices], []
    validation_count = max(1, int(len(texts) * validation_split))
    validation_count = min(validation_count, len(texts) - 1)
    validation_indices = set(indices[:validation_count])
    train = [text for i, text in enumerate(texts) if i not in validation_indices]
    validation = [text for i, text in enumerate(texts) if i in validation_indices]
    return train, validation


def _tokenize_corpus(texts: list[str], tokenizer: CharTokenizer, context: int) -> torch.Tensor:
    ids: list[int] = []
    separator = tokenizer.stoi.get("\n", 0)
    for text in texts:
        encoded = tokenizer.encode(text)
        if len(encoded) >= 2:
            ids.extend(encoded)
            ids.append(separator)
    if len(ids) <= context:
        raise ValueError("dataset is shorter than the requested context")
    return torch.tensor(ids, dtype=torch.long)


def _sample_batch(tokens: torch.Tensor, batch_size: int, context: int, device: torch.device):
    maximum = len(tokens) - context - 1
    if maximum <= 0:
        raise ValueError("token corpus is shorter than context + 2")
    starts = torch.randint(0, maximum, (batch_size,))
    x = torch.stack([tokens[i : i + context] for i in starts])
    y = torch.stack([tokens[i + 1 : i + context + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, tokens: torch.Tensor, batch_size: int, context: int, device: torch.device) -> float | None:
    """Estimate validation loss with deterministic contiguous windows."""
    if len(tokens) <= context + 1:
        return None
    model.eval()
    window_count = len(tokens) - context - 1
    starts = list(range(window_count))
    # Cap validation work for very large corpora while remaining deterministic.
    if len(starts) > 256:
        stride = max(1, len(starts) // 256)
        starts = starts[::stride][:256]
    losses = []
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        x = torch.stack([tokens[i : i + context] for i in batch_starts]).to(device)
        y = torch.stack([tokens[i + 1 : i + context + 1] for i in batch_starts]).to(device)
        losses.append(float(model.loss(x, y).item()))
    return sum(losses) / len(losses) if losses else None


class TrainingPipeline:
    """Train, validate, checkpoint, and resume a TARA language model."""

    def __init__(self, config: TrainingConfig | None = None, device: str | None = None) -> None:
        self.config = config or TrainingConfig()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable")

    def _new_state(self, texts: list[str]):
        train_texts, validation_texts = split_texts(
            texts, self.config.validation_split, self.config.seed
        )
        tokenizer = CharTokenizer("\n".join(train_texts))
        train_tokens = _tokenize_corpus(train_texts, tokenizer, self.config.context)
        validation_tokens = (
            _tokenize_corpus(validation_texts, tokenizer, self.config.context)
            if validation_texts
            else None
        )
        model = FastTinyLanguageModel(
            **self.config.model_config(tokenizer.vocab_size),
            seed=self.config.seed,
        ).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr)
        return model, tokenizer, optimizer, train_tokens, validation_tokens

    @staticmethod
    def _load_tokenizer(payload: dict) -> CharTokenizer:
        data = payload.get("tokenizer")
        if not isinstance(data, dict) or not isinstance(data.get("itos"), list):
            raise ValueError("checkpoint tokenizer is missing or invalid")
        tokenizer = CharTokenizer("a")
        tokenizer.itos = list(data["itos"])
        tokenizer.stoi = dict(data["stoi"])
        return tokenizer

    def _resume_state(self, checkpoint_path: Path, texts: list[str]):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported TARA training checkpoint format")
        checkpoint_fingerprint = payload.get("dataset_fingerprint")
        current_fingerprint = _fingerprint(texts)
        if checkpoint_fingerprint != current_fingerprint:
            raise ValueError(
                "dataset fingerprint differs from checkpoint; use the original dataset "
                "or start a fresh training run"
            )
        tokenizer = self._load_tokenizer(payload)
        model_config = payload.get("model_config")
        if model_config != self.config.model_config(tokenizer.vocab_size):
            raise ValueError("checkpoint model configuration does not match training config")
        model = FastTinyLanguageModel(**model_config, seed=self.config.seed).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        model.train()
        train_texts, validation_texts = split_texts(
            texts, self.config.validation_split, self.config.seed
        )
        train_tokens = _tokenize_corpus(train_texts, tokenizer, self.config.context)
        validation_tokens = (
            _tokenize_corpus(validation_texts, tokenizer, self.config.context)
            if validation_texts
            else None
        )
        step = payload.get("step")
        if not isinstance(step, int) or step < 0:
            raise ValueError("checkpoint step is invalid")
        return model, tokenizer, optimizer, train_tokens, validation_tokens, step

    def _save(self, path: Path, model, tokenizer, optimizer, step, train_loss, validation_loss, fingerprint):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "step": step,
            "model_state": model.state_dict(),
            "model_config": self.config.model_config(tokenizer.vocab_size),
            "tokenizer": {"itos": tokenizer.itos, "stoi": tokenizer.stoi},
            "optimizer_state": optimizer.state_dict(),
            "metrics": {
                "train_loss": float(train_loss),
                "validation_loss": None if validation_loss is None else float(validation_loss),
            },
            "dataset_fingerprint": fingerprint,
            "config": asdict(self.config),
        }
        torch.save(payload, path)

    def train(self, data: str | Path, output: str | Path, resume: str | Path | None = None) -> TrainingSummary:
        """Train for ``config.steps`` additional steps and save the final checkpoint."""
        texts = load_training_texts(data)
        fingerprint = _fingerprint(texts)
        output_path = Path(output)
        if resume is None:
            model, tokenizer, optimizer, train_tokens, validation_tokens = self._new_state(texts)
            start_step = 0
        else:
            model, tokenizer, optimizer, train_tokens, validation_tokens, start_step = self._resume_state(
                Path(resume), texts
            )

        random.seed(self.config.seed + start_step)
        torch.manual_seed(self.config.seed + start_step)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed + start_step)

        last_train_loss = float("nan")
        last_validation_loss = None
        model.train()
        for local_step in range(1, self.config.steps + 1):
            step = start_step + local_step
            x, y = _sample_batch(train_tokens, self.config.batch_size, self.config.context, self.device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            optimizer.step()
            last_train_loss = float(loss.item())

            if local_step == 1 or local_step % self.config.log_every == 0 or local_step == self.config.steps:
                last_validation_loss = evaluate(
                    model, validation_tokens, self.config.batch_size, self.config.context, self.device
                ) if validation_tokens is not None else None
                print(
                    f"step={step:5d} train_loss={last_train_loss:.4f} "
                    f"val_loss={last_validation_loss:.4f} device={self.device}"
                    if last_validation_loss is not None
                    else f"step={step:5d} train_loss={last_train_loss:.4f} device={self.device}"
                )

            if self.config.checkpoint_every and step % self.config.checkpoint_every == 0:
                self._save(
                    output_path, model, tokenizer, optimizer, step,
                    last_train_loss, last_validation_loss, fingerprint,
                )

        self._save(
            output_path, model, tokenizer, optimizer, start_step + self.config.steps,
            last_train_loss, last_validation_loss, fingerprint,
        )
        return TrainingSummary(
            checkpoint=str(output_path),
            start_step=start_step,
            final_step=start_step + self.config.steps,
            train_loss=last_train_loss,
            validation_loss=last_validation_loss,
            device=str(self.device),
            dataset_fingerprint=fingerprint,
        )
