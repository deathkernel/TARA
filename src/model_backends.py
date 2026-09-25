"""Optional Hugging Face model backends for TARA.

The native tiny model remains the default. This module adds a common lazy-loading
adapter for compatible text-generation checkpoints so TARA can use Llama, Code
Llama, Gemma, and Mistral without copying their source trees into the project.

Model weights are never bundled by TARA. Users must install Transformers and
obtain access to any gated model according to its publisher's terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    family: str
    source: str
    license_note: str


MODEL_REGISTRY = {
    "llama": ModelSpec(
        "llama", "meta-llama/Llama-3.2-3B-Instruct", "Llama",
        "Meta / Hugging Face", "Meta Llama 3.2 license and use policy apply.",
    ),
    "codellama": ModelSpec(
        "codellama", "codellama/CodeLlama-7b-Instruct-hf", "Code Llama",
        "Meta / Hugging Face", "Code Llama license and use policy apply.",
    ),
    "gemma": ModelSpec(
        "gemma", "google/gemma-3-4b-it", "Gemma",
        "Google DeepMind / Hugging Face", "Gemma terms and use policy apply.",
    ),
    "mistral": ModelSpec(
        "mistral", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "Mistral",
        "Mistral AI / Hugging Face", "Apache-2.0 model card terms apply to this checkpoint.",
    ),
}


class TransformersBackend:
    """Lazy Hugging Face text-generation backend used by TARA."""

    def __init__(self, model_id: str, *, device: str = "auto", dtype: str = "auto"):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.tokenizer = None
        self.model = None

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers backend requires optional dependencies. "
                "Install with: pip install -r requirements-models.txt"
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.device == "auto":
            kwargs["device_map"] = "auto"
        if self.dtype != "auto":
            import torch
            kwargs["torch_dtype"] = getattr(torch, self.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if self.device != "auto":
            self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        self._load()
        assert self.tokenizer is not None and self.model is not None
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = output[:, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def create_backend(name: str, *, model_id: str | None = None, device: str = "auto") -> TransformersBackend:
    key = name.lower()
    if key not in MODEL_REGISTRY and not model_id:
        raise ValueError(
            f"Unknown model backend '{name}'. Available: {', '.join(MODEL_REGISTRY)}"
        )
    spec = MODEL_REGISTRY.get(key)
    return TransformersBackend(model_id or spec.model_id, device=device)


def registry_summary() -> tuple[ModelSpec, ...]:
    return tuple(MODEL_REGISTRY.values())
