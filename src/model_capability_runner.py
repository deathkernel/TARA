"""Run TARA's real language model against deterministic capability cases.

The runner adapts the model's text-generation interface to the capability
benchmark. It intentionally performs inference only; it never trains or
promotes a checkpoint. Evaluation uses greedy decoding so repeated runs are
comparable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

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
    """Fingerprint model architecture and tokenizer configuration, not weights."""
    config = getattr(model, "config", None)
    payload = {
        "model_type": type(model).__name__,
        "tokenizer_type": type(tokenizer).__name__,
        "model_config": getattr(config, "__dict__", repr(config)),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "tokenizer_vocab": getattr(tokenizer, "itos", None),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class ModelCapabilityRunner:
    """Evaluate an actual TARA checkpoint through deterministic greedy decoding."""

    def __init__(self, model: Any, tokenizer: Any, *, max_new_tokens: int = 32):
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("prompt must encode to at least one token")
        token_ids = list(prompt_ids)
        for _ in range(self.max_new_tokens):
            logits = self.model.forward_numeric(token_ids)[-1]
            token_id = max(range(len(logits)), key=logits.__getitem__)
            token_ids.append(token_id)
        return self.tokenizer.decode(token_ids[len(prompt_ids):])

    def evaluate(self, cases: tuple[BenchmarkCase, ...], *, name: str = "tara-real-model") -> ModelEvaluation:
        benchmark = IntelligenceBenchmark(name, cases)
        report = benchmark.run(self.generate)
        model_id = model_fingerprint(self.model, self.tokenizer)
        fingerprint = hashlib.sha256(
            json.dumps({"model": model_id, "benchmark": report.fingerprint}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return ModelEvaluation("in-memory", report, model_id, fingerprint)


def evaluate_checkpoint(
    path: str | Path,
    cases: tuple[BenchmarkCase, ...],
    *,
    name: str = "tara-real-model",
    max_new_tokens: int = 32,
) -> ModelEvaluation:
    from .model_runtime import load_checkpoint
    model, tokenizer = load_checkpoint(path)
    evaluation = ModelCapabilityRunner(model, tokenizer, max_new_tokens=max_new_tokens).evaluate(cases, name=name)
    return ModelEvaluation(str(path), evaluation.benchmark, evaluation.model_fingerprint, evaluation.fingerprint)


def write_evaluation(evaluation: ModelEvaluation, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evaluation.as_dict(), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return destination
