"""Compatibility wrapper; canonical implementation lives in src.tara_mind.core.transformer."""

from src.tara_mind.core.transformer import CausalSelfAttention, FastTinyLanguageModel, RotaryEmbedding, SwiGLU, TransformerBlock

__all__ = ["CausalSelfAttention", "FastTinyLanguageModel", "RotaryEmbedding", "SwiGLU", "TransformerBlock"]
