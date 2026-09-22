"""Checkpoint-backed adapter for TARA's algorithm language model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .generation import ModelCandidateGenerator


class TARAAlgorithmModel:
    """Lazy, CPU-safe wrapper around a trained algorithm-LM checkpoint."""

    def __init__(self, checkpoint: str | Path, *, max_new_tokens: int = 512) -> None:
        self.checkpoint = Path(checkpoint)
        self.max_new_tokens = max_new_tokens
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def load(self) -> "TARAAlgorithmModel":
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"TARA model checkpoint not found: {self.checkpoint}")
        from src.model_runtime import load_checkpoint
        self._model, self._tokenizer = load_checkpoint(self.checkpoint)
        return self

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def generate(self, prompt: str, *, max_new_tokens: int | None = None, temperature: float = 0.8) -> str:
        if not self.loaded:
            self.load()
        assert self._model is not None and self._tokenizer is not None
        from src.model_runtime import generate_text
        return generate_text(
            self._model,
            self._tokenizer,
            prompt,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=temperature,
        )

    def candidate_generator(self, *, temperatures: tuple[float, ...] = (0.65, 0.85, 1.05)) -> ModelCandidateGenerator:
        return ModelCandidateGenerator(self.generate, max_new_tokens=self.max_new_tokens, temperatures=temperatures)
