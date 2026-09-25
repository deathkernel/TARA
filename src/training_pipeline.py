"""End-to-end PyTorch training pipeline for TARA's learned language core."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

from src.tokenizer import BPETokenizer, CharTokenizer
from src.torch_language_model import FastTinyLanguageModel
from src.training_hardening import EarlyStopping, ExperimentTracker, TrainingControls, TrainingMetric, WarmupCosineScheduler

CHECKPOINT_FORMAT_VERSION = 5

@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 1000
    batch_size: int = 32
    context: int = 128
    embedding_dim: int = 64
    ff_dim: int = 128
    heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    tie_embeddings: bool = True
    tokenizer: str = "bpe"
    vocab_size: int = 2048
    lr: float = 3e-4
    validation_split: float = 0.1
    seed: int = 42
    log_every: int = 100
    checkpoint_every: int = 0
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    target_validation_accuracy: float | None = None

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0 or self.context <= 0:
            raise ValueError("steps, batch_size, and context must be positive")
        if self.embedding_dim <= 0 or self.ff_dim <= 0 or self.heads <= 0 or self.num_layers <= 0:
            raise ValueError("model dimensions and num_layers must be positive")
        if self.embedding_dim % self.heads != 0:
            raise ValueError("embedding_dim must be divisible by heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.tokenizer not in {"char", "bpe"}:
            raise ValueError("tokenizer must be char or bpe")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError("validation_split must be in [0, 1)")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.checkpoint_every < 0 or self.grad_clip <= 0:
            raise ValueError("checkpoint_every must be non-negative and grad_clip positive")
        if self.target_validation_accuracy is not None and not 0.0 < self.target_validation_accuracy <= 1.0:
            raise ValueError("target_validation_accuracy must be in (0, 1]")
        TrainingControls(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            min_lr_ratio=self.min_lr_ratio,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
        )

    def model_config(self, vocab_size: int) -> dict[str, int | float | bool]:
        return {"vocab_size": vocab_size, "embedding_dim": self.embedding_dim, "ff_dim": self.ff_dim, "num_heads": self.heads, "max_context": self.context, "num_layers": self.num_layers, "dropout": self.dropout, "tie_embeddings": self.tie_embeddings}

    def hardening_config(self) -> dict[str, int | float]:
        return {"gradient_accumulation_steps": self.gradient_accumulation_steps, "warmup_steps": self.warmup_steps, "min_lr_ratio": self.min_lr_ratio, "patience": self.early_stopping_patience, "min_delta": self.early_stopping_min_delta}

@dataclass(frozen=True)
class TrainingSummary:
    checkpoint: str
    start_step: int
    final_step: int
    train_loss: float
    validation_loss: float | None
    validation_accuracy: float | None
    device: str
    dataset_fingerprint: str
    stopped_early: bool = False
    metrics_path: str | None = None

def _fingerprint(texts: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()

def _record_to_text(item: object) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        raise ValueError("dataset record must be a string or object")
    if "problem" in item and "solution" in item:
        return "Problem: " + str(item["problem"]) + "\nApproach: " + str(item["solution"]) + "\nTests: " + str(item.get("tests", ""))
    if "prompt" in item and "solution" in item:
        return "Prompt: " + str(item["prompt"]) + "\nAnswer: " + str(item["solution"]) + "\nTests: " + str(item.get("tests", ""))
    value = item.get("text", item.get("content", item.get("prompt")))
    if value is None:
        raise ValueError("dataset record has no text/content/prompt field")
    return str(value).strip()

def load_training_texts(path: str | Path) -> list[str]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    raw = source.read_text(encoding="utf-8")
    if source.suffix.lower() in {".txt", ".text"}:
        records = [line.strip() for line in raw.splitlines() if line.strip()]
    else:
        lines = [line for line in raw.splitlines() if line.strip()]
        try:
            payload = json.loads(raw)
            items = payload.get("records", payload) if isinstance(payload, dict) else payload
            iterable = items if isinstance(items, list) else [items]
        except json.JSONDecodeError:
            iterable = [json.loads(line) for line in lines]
        records = [_record_to_text(item) for item in iterable]
    records = [text for text in records if text]
    if not records:
        raise ValueError("training dataset is empty")
    return records

def split_texts(texts: list[str], validation_split: float, seed: int) -> tuple[list[str], list[str]]:
    if not texts:
        raise ValueError("texts must not be empty")
    indices = list(range(len(texts)))
    random.Random(seed).shuffle(indices)
    if validation_split == 0 or len(texts) < 2:
        return [texts[i] for i in indices], []
    validation_count = min(max(1, int(len(texts) * validation_split)), len(texts) - 1)
    validation_indices = set(indices[:validation_count])
    return ([text for i, text in enumerate(texts) if i not in validation_indices], [text for i, text in enumerate(texts) if i in validation_indices])

def _tokenize_corpus(texts: list[str], tokenizer: CharTokenizer | BPETokenizer, context: int) -> list[torch.Tensor]:
    """Encode each record separately so training never crosses a record boundary."""
    sequences: list[torch.Tensor] = []
    for text in texts:
        encoded = tokenizer.encode(text)
        if len(encoded) >= context + 1:
            sequences.append(torch.tensor(encoded, dtype=torch.long))
    if not sequences:
        raise ValueError("dataset has no record long enough for the requested context")
    return sequences

def _sample_batch(sequences: list[torch.Tensor], batch_size: int, context: int, device: torch.device):
    """Sample fixed-length windows entirely inside individual records."""
    if not sequences:
        raise ValueError("token sequence collection is empty")
    valid = [index for index, sequence in enumerate(sequences) if len(sequence) > context]
    if not valid:
        raise ValueError("dataset has no record long enough for the requested context")
    sequence_indices = torch.randint(0, len(valid), (batch_size,)).tolist()
    rows = []
    targets = []
    for choice in sequence_indices:
        sequence = sequences[valid[choice]]
        start = random.randrange(0, len(sequence) - context)
        rows.append(sequence[start:start + context])
        targets.append(sequence[start + 1:start + context + 1])
    x = torch.stack(rows).to(device, non_blocking=True)
    y = torch.stack(targets).to(device, non_blocking=True)
    return x, y

@torch.no_grad()
def evaluate_metrics(model, sequences: list[torch.Tensor] | None, batch_size: int, context: int, device: torch.device) -> tuple[float, float] | None:
    """Measure held-out next-token loss and exact token accuracy without crossing records."""
    if not sequences:
        return None
    windows: list[tuple[int, int]] = []
    for sequence_index, sequence in enumerate(sequences):
        windows.extend((sequence_index, start) for start in range(max(0, len(sequence) - context)))
    if not windows:
        return None
    if len(windows) > 256:
        stride = max(1, len(windows) // 256)
        windows = windows[::stride][:256]
    model.eval()
    losses = []
    correct = 0
    total = 0
    for offset in range(0, len(windows), batch_size):
        batch = windows[offset:offset + batch_size]
        x = torch.stack([sequences[i][start:start + context] for i, start in batch]).to(device)
        y = torch.stack([sequences[i][start + 1:start + context + 1] for i, start in batch]).to(device)
        logits = model(x)
        losses.append(float(torch.nn.functional.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1)).item()))
        correct += int((logits.argmax(dim=-1) == y).sum().item())
        total += int(y.numel())
    return (sum(losses) / len(losses), correct / total) if losses and total else None

@torch.no_grad()
def evaluate(model, tokens: torch.Tensor | None, batch_size: int, context: int, device: torch.device) -> float | None:
    metrics = evaluate_metrics(model, tokens, batch_size, context, device)
    return None if metrics is None else metrics[0]

class TrainingPipeline:
    """Train, validate, checkpoint, resume, and track a TARA language model."""
    def __init__(self, config: TrainingConfig | None = None, device: str | None = None) -> None:
        self.config = config or TrainingConfig()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable")

    def _new_state(self, texts: list[str]):
        train_texts, validation_texts = split_texts(texts, self.config.validation_split, self.config.seed)
        corpus = "\n".join(train_texts)
        print(f"Preparing tokenizer ({self.config.tokenizer})...")
        tokenizer = (BPETokenizer(corpus, vocab_size=self.config.vocab_size) if self.config.tokenizer == "bpe" else CharTokenizer(corpus))
        print(f"Tokenizer ready: vocab={tokenizer.vocab_size}")
        print("Encoding training corpus...")
        train_tokens = _tokenize_corpus(train_texts, tokenizer, self.config.context)
        validation_tokens = _tokenize_corpus(validation_texts, tokenizer, self.config.context) if validation_texts else None
        print(f"Training tokens: {len(train_tokens):,}; validation tokens: {0 if validation_tokens is None else len(validation_tokens):,}")
        model = FastTinyLanguageModel(**self.config.model_config(tokenizer.vocab_size), seed=self.config.seed).to(self.device)
        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr, fused=self.device.type == "cuda")
        except (TypeError, RuntimeError):
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr)
        return model, tokenizer, optimizer, train_tokens, validation_tokens

    @staticmethod
    def _load_tokenizer(payload: dict) -> CharTokenizer | BPETokenizer:
        data = payload.get("tokenizer")
        if not isinstance(data, dict) or not isinstance(data.get("itos"), list):
            raise ValueError("checkpoint tokenizer is missing or invalid")
        kind = data.get("type", "char")
        if kind == "char":
            tokenizer = CharTokenizer("a")
            tokenizer.itos = list(data["itos"])
            tokenizer.stoi = dict(data["stoi"])
            return tokenizer
        if kind == "bpe":
            tokenizer = BPETokenizer("a", vocab_size=max(2, len(data["itos"])))
            tokenizer.itos = list(data["itos"])
            tokenizer.stoi = dict(data["stoi"])
            tokenizer.merges = [tuple(pair) for pair in data.get("merges", [])]
            tokenizer._merge_ranks = {pair: index for index, pair in enumerate(tokenizer.merges)}
            return tokenizer
        raise ValueError("unsupported checkpoint tokenizer type")

    def _resume_state(self, checkpoint_path: Path, texts: list[str]):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported TARA training checkpoint format; retrain or migrate the checkpoint")
        if payload.get("dataset_fingerprint") != _fingerprint(texts):
            raise ValueError("dataset fingerprint differs from checkpoint; use the original dataset or start a fresh training run")
        if payload.get("hardening_config") != self.config.hardening_config():
            raise ValueError("checkpoint training controls do not match training config")
        tokenizer = self._load_tokenizer(payload)
        model_config = payload.get("model_config")
        if model_config != self.config.model_config(tokenizer.vocab_size):
            raise ValueError("checkpoint model configuration does not match training config")
        model = FastTinyLanguageModel(**model_config, seed=self.config.seed).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        train_texts, validation_texts = split_texts(texts, self.config.validation_split, self.config.seed)
        train_tokens = _tokenize_corpus(train_texts, tokenizer, self.config.context)
        validation_tokens = _tokenize_corpus(validation_texts, tokenizer, self.config.context) if validation_texts else None
        step = payload.get("step")
        if not isinstance(step, int) or step < 0:
            raise ValueError("checkpoint step is invalid")
        return model, tokenizer, optimizer, train_tokens, validation_tokens, step, payload

    def _save(self, path: Path, model, tokenizer, optimizer, step, train_loss, validation_loss, fingerprint, scheduler, early_stopping, tracker, validation_accuracy=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "step": step,
            "model_state": model.state_dict(),
            "model_config": self.config.model_config(tokenizer.vocab_size),
            "tokenizer": {
                "type": "bpe" if isinstance(tokenizer, BPETokenizer) else "char",
                "itos": tokenizer.itos,
                "stoi": tokenizer.stoi,
                "merges": tokenizer.merges if isinstance(tokenizer, BPETokenizer) else [],
            },
            "optimizer_state": optimizer.state_dict(),
            "metrics": {"train_loss": float(train_loss), "validation_loss": None if validation_loss is None else float(validation_loss), "validation_accuracy": None if validation_accuracy is None else float(validation_accuracy)},
            "dataset_fingerprint": fingerprint,
            "config": asdict(self.config),
            "hardening_config": self.config.hardening_config(),
            "scheduler": {"total_steps": scheduler.total_steps, "warmup_steps": scheduler.warmup_steps, "min_lr_ratio": scheduler.min_lr_ratio, "optimizer_steps": step},
            "early_stopping": {"best": early_stopping.best, "bad_steps": early_stopping.bad_steps},
            "metrics_fingerprint": tracker.fingerprint(),
        }, path)

    def train(self, data: str | Path, output: str | Path, resume: str | Path | None = None, metrics_path: str | Path | None = None) -> TrainingSummary:
        print(f"Loading dataset: {data}")
        texts = load_training_texts(data)
        print(f"Loaded {len(texts):,} records")
        fingerprint = _fingerprint(texts)
        output_path = Path(output)
        tracker_path = Path(metrics_path) if metrics_path is not None else output_path.with_suffix(output_path.suffix + ".metrics.jsonl")
        tracker = ExperimentTracker(tracker_path)
        if resume is None:
            model, tokenizer, optimizer, train_tokens, validation_tokens = self._new_state(texts)
            start_step = 0
            early_stopping = EarlyStopping(self.config.early_stopping_patience, self.config.early_stopping_min_delta)
        else:
            model, tokenizer, optimizer, train_tokens, validation_tokens, start_step, payload = self._resume_state(Path(resume), texts)
            state = payload.get("early_stopping", {})
            early_stopping = EarlyStopping(self.config.early_stopping_patience, self.config.early_stopping_min_delta)
            early_stopping.best = state.get("best")
            early_stopping.bad_steps = int(state.get("bad_steps", 0))
        if self.device.type == "cuda":
            train_tokens = [tokens.to(self.device) for tokens in train_tokens]
            if validation_tokens is not None:
                validation_tokens = [tokens.to(self.device) for tokens in validation_tokens]
        total_steps = start_step + self.config.steps
        scheduler = WarmupCosineScheduler(total_steps=max(1, total_steps), warmup_steps=min(self.config.warmup_steps, max(1, total_steps)), min_lr_ratio=self.config.min_lr_ratio)
        random.seed(self.config.seed + start_step)
        torch.manual_seed(self.config.seed + start_step)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed + start_step)
        last_train_loss = float("nan")
        last_validation_loss = None
        last_validation_accuracy = None
        stopped_early = False
        model.train()
        optimizer.zero_grad(set_to_none=True)
        update_step = start_step
        while update_step < total_steps:
            accumulated_loss = 0.0
            for _ in range(self.config.gradient_accumulation_steps):
                x, y = _sample_batch(train_tokens, self.config.batch_size, self.config.context, self.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                    loss = model.loss(x, y)
                (loss / self.config.gradient_accumulation_steps).backward()
                accumulated_loss += float(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            update_step += 1
            current_lr = self.config.lr * scheduler.multiplier(update_step - 1)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            last_train_loss = accumulated_loss / self.config.gradient_accumulation_steps
            should_log = update_step == start_step + 1 or update_step % self.config.log_every == 0 or update_step == total_steps
            if should_log:
                model.eval()
                validation_metrics = evaluate_metrics(model, validation_tokens, self.config.batch_size, self.config.context, self.device)
                last_validation_loss = None if validation_metrics is None else validation_metrics[0]
                last_validation_accuracy = None if validation_metrics is None else validation_metrics[1]
                tracker.log(TrainingMetric(update_step, last_train_loss, last_validation_loss, current_lr, last_validation_accuracy))
                decision = early_stopping.update(last_validation_loss) if last_validation_loss is not None else None
                print(f"step={update_step:5d} train_loss={last_train_loss:.4f} " + (f"val_loss={last_validation_loss:.4f} val_accuracy={last_validation_accuracy:.2%} " if last_validation_loss is not None else "") + f"lr={current_lr:.6g} device={self.device}")
                if decision is not None and decision.improved:
                    self._save(output_path, model, tokenizer, optimizer, update_step, last_train_loss, last_validation_loss, fingerprint, scheduler, early_stopping, tracker, last_validation_accuracy)
                if self.config.target_validation_accuracy is not None and last_validation_accuracy is not None and last_validation_accuracy >= self.config.target_validation_accuracy:
                    stopped_early = True
                    break
                if decision is not None and decision.should_stop:
                    stopped_early = True
                    break
                model.train()
            if self.config.checkpoint_every and update_step % self.config.checkpoint_every == 0:
                self._save(output_path, model, tokenizer, optimizer, update_step, last_train_loss, last_validation_loss, fingerprint, scheduler, early_stopping, tracker, last_validation_accuracy)
        self._save(output_path, model, tokenizer, optimizer, update_step, last_train_loss, last_validation_loss, fingerprint, scheduler, early_stopping, tracker, last_validation_accuracy)
        return TrainingSummary(str(output_path), start_step, update_step, last_train_loss, last_validation_loss, last_validation_accuracy, str(self.device), fingerprint, stopped_early, str(tracker_path))
