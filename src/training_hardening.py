"""Phase 37.3 training controls: accumulation, scheduling, early stopping, tracking."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingControls:
    """Validated controls independent of the neural backend."""
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    patience: int = 5
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0.0 < self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in (0, 1]")
        if self.patience < 0 or self.min_delta < 0:
            raise ValueError("patience and min_delta must be non-negative")


class WarmupCosineScheduler:
    """Deterministic warmup + cosine decay multiplier."""
    def __init__(self, total_steps: int, warmup_steps: int = 0, min_lr_ratio: float = 0.1):
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio

    def multiplier(self, step: int) -> float:
        import math
        if step < 0:
            raise ValueError("step must be non-negative")
        if self.warmup_steps and step < self.warmup_steps:
            return max(1e-12, (step + 1) / self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True)
class EarlyStopDecision:
    improved: bool
    should_stop: bool
    best_value: float | None
    bad_steps: int


class EarlyStopping:
    """Validation-loss early stopping with explicit minimum improvement."""
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best: float | None = None
        self.bad_steps = 0

    def update(self, value: float) -> EarlyStopDecision:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("validation metric must be finite")
        improved = self.best is None or value < self.best - self.min_delta
        if improved:
            self.best = value
            self.bad_steps = 0
        else:
            self.bad_steps += 1
        return EarlyStopDecision(improved, self.bad_steps > self.patience, self.best, self.bad_steps)


@dataclass(frozen=True)
class TrainingMetric:
    step: int
    train_loss: float
    validation_loss: float | None
    learning_rate: float


class ExperimentTracker:
    """Append-only JSONL tracker with a stable experiment fingerprint."""
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def log(self, metric: TrainingMetric) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(metric), sort_keys=True) + "\n")

    def metrics(self) -> list[TrainingMetric]:
        if not self.path.exists():
            return []
        result = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result.append(TrainingMetric(**json.loads(line)))
        return result

    def fingerprint(self) -> str:
        payload = self.path.read_bytes() if self.path.exists() else b""
        return hashlib.sha256(payload).hexdigest()
