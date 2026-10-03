"""Lazy capability adapters for TARA's upstream ecosystem."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdapterStatus:
    id: str
    installed: bool
    version: str | None
    error: str | None = None


_PACKAGE_IMPORTS = {
    "tensorflow": "tensorflow",
    "jax": "jax",
    "gemma": "gemma",
    "onnxruntime": "onnxruntime",
    "onnxruntime-genai": "onnxruntime_genai",
    "semantic-kernel": "semantic_kernel",
    "whisper": "whisper",
    "openai-python": "openai",
    "tiktoken": "tiktoken",
    "phi": "transformers",
}


def status(adapter_id: str) -> AdapterStatus:
    module_name = _PACKAGE_IMPORTS.get(adapter_id)
    if module_name is None:
        return AdapterStatus(adapter_id, False, None, "metadata-only integration")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return AdapterStatus(adapter_id, False, None, f"{type(exc).__name__}: {exc}")
    version = getattr(module, "__version__", None)
    return AdapterStatus(adapter_id, True, str(version) if version else None)


def installed_capabilities() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for adapter_id in _PACKAGE_IMPORTS:
        item = status(adapter_id)
        if item.installed:
            result[adapter_id] = {"version": item.version, "status": "ready"}
    return result


def create_openai_client(**kwargs):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the optional OpenAI integration first.") from exc
    return OpenAI(**kwargs)


def openai_chat(
    messages: list[dict[str, str]],
    *,
    model: str = "gpt-oss-20b",
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Use the OpenAI SDK for hosted OpenAI or OpenAI-compatible local models.

    base_url can point at a local gpt-oss-compatible server. Secrets are always
    supplied by the caller/environment and are never persisted by this adapter.
    """
    kwargs: dict[str, Any] = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    client = create_openai_client(**kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return str(response.choices[0].message.content or "").strip()


def load_whisper(model_name: str = "base"):
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError("Install openai-whisper to enable speech input.") from exc
    return whisper.load_model(model_name)


def load_onnx_model(path: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install onnxruntime to enable ONNX inference.") from exc
    return ort.InferenceSession(path)


def load_jax():
    try:
        return importlib.import_module("jax")
    except ImportError as exc:
        raise RuntimeError("Install jax to enable the JAX backend.") from exc


def load_tensorflow():
    try:
        return importlib.import_module("tensorflow")
    except ImportError as exc:
        raise RuntimeError("Install tensorflow to enable the TensorFlow backend.") from exc


def load_semantic_kernel():
    try:
        return importlib.import_module("semantic_kernel")
    except ImportError as exc:
        raise RuntimeError("Install semantic-kernel for the optional orchestration bridge.") from exc


def encode_tiktoken(text: str, encoding: str = "o200k_base") -> list[int]:
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("Install tiktoken to enable OpenAI-compatible token accounting.") from exc
    return tiktoken.get_encoding(encoding).encode(text)



def load_onnx_genai_model(path: str):
    try:
        import onnxruntime_genai as og
    except ImportError as exc:
        raise RuntimeError("Install onnxruntime-genai for optimized local LLM inference.") from exc
    return og.Model(path)
