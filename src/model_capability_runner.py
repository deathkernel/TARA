"""Run TARA's real language model against deterministic capability cases.

The runner adapts a checkpoint to the capability benchmark. It intentionally
performs inference only; it never trains or promotes a checkpoint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    observations: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_fingerprint(model: Any, tokenizer: Any) -> str:
    """Fingerprint model architecture and tokenizer configuration."""
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
        import torch

        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("prompt must encode to at least one token")
        token_ids = list(prompt_ids)
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                context_ids = token_ids[-self.model.max_context :]
                x = torch.tensor([context_ids], dtype=torch.long)
                logits = self.model(x)[0, -1]
                token_ids.append(int(torch.argmax(logits).item()))
        return self.tokenizer.decode(token_ids[len(prompt_ids):])

    def evaluate(self, cases: tuple[BenchmarkCase, ...], *, name: str = "tara-real-model") -> ModelEvaluation:
        benchmark = IntelligenceBenchmark(name, cases)
        observations: dict[str, str] = {}
        for case in cases:
            try:
                observations[case.case_id] = self.generate(case.prompt)
            except Exception as exc:
                observations[case.case_id] = f"<execution-error:{type(exc).__name__}: {exc}>"

        report = benchmark.run(observations.__getitem__)
        model_id = model_fingerprint(self.model, self.tokenizer)
        fingerprint = hashlib.sha256(
            json.dumps(
                {"model": model_id, "benchmark": report.fingerprint, "observations": observations},
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        return ModelEvaluation("in-memory", report, model_id, fingerprint, observations)


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
    return ModelEvaluation(str(path), evaluation.benchmark, evaluation.model_fingerprint, evaluation.fingerprint, evaluation.observations)


def write_evaluation(evaluation: ModelEvaluation, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evaluation.as_dict(), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return destination
