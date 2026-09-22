"""Run TARA's real language model against deterministic capability cases.

The runner adapts the model's text-generation interface to the capability
benchmark. It intentionally performs inference only; it never trains or
promotes a checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .intelligence_benchmark import BenchmarkCase, BenchmarkReport, IntelligenceBenchmark


@dataclass(frozen=True)
class ModelEvaluation:
    checkpoint: str
    benchmark: BenchmarkReport
    model_fingerprint: str
    fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_fingerprint(model: Any, tokenizer: Any) -> str:
    """Create a stable fingerprint from model/tokenizer configuration."""
    model_state = getattr(model, "state_dict", lambda: {})()
    parts: list[str] = [type(model).__name__, type(tokenizer).__name__]
    for key in sorted(model_state):
        value = model_state[key]
        shape = tuple(getattr(value, "shape", ()))
        parts.append(f"{key}:{shape}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


class ModelCapabilityRunner:
    """Evaluate an actual TARA checkpoint through its public generation API."""

    def __init__(self, model: Any, tokenizer: Any, *, max_new_tokens: int = 32, temperature: float = 0.0):
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        token_ids = self.tokenizer.encode(prompt)
        if not token_ids:
            raise ValueError("prompt must encode to at least one token")
        for _ in range(self.max_new_tokens):
            logits = self.model.forward_numeric(token_ids)[-1]
            if self.temperature == 0.0:
                token_id = max(range(len(logits)), key=logits.__getitem__)
            else:
                import math
                probs = [math.exp(float(x) / self.temperature - max(logits) / self.temperature) for x in logits]
                total = sum(probs)
                token_id = max(range(len(probs)), key=probs.__getitem__) if total <= 0 else max(range(len(probs)), key=probs.__getitem__)
            token_ids.append(token_id)
        return self.tokenizer.decode(token_ids[len(self.tokenizer.encode(prompt)):])

    def evaluate(self, cases: tuple[BenchmarkCase, ...], *, name: str = "tara-real-model") -> ModelEvaluation:
        benchmark = IntelligenceBenchmark(name, cases)
        report = benchmark.run(self.generate)
        fingerprint = hashlib.sha256(
            json.dumps({"model": model_fingerprint(self.model, self.tokenizer), "benchmark": report.fingerprint}, sort_keys=True).encode()
        ).hexdigest()
        return ModelEvaluation("in-memory", report, model_fingerprint(self.model, self.tokenizer), fingerprint)


def evaluate_checkpoint(path: str | Path, cases: tuple[BenchmarkCase, ...], *, name: str = "tara-real-model", max_new_tokens: int = 32) -> ModelEvaluation:
    from .model_runtime import load_checkpoint
    model, tokenizer = load_checkpoint(path)
    evaluation = ModelCapabilityRunner(model, tokenizer, max_new_tokens=max_new_tokens).evaluate(cases, name=name)
    return ModelEvaluation(str(path), evaluation.benchmark, evaluation.model_fingerprint, evaluation.fingerprint)


def write_evaluation(evaluation: ModelEvaluation, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evaluation.as_dict(), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return destination
