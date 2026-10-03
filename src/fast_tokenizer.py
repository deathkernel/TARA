"""Fast Rust-backed BPE tokenizer for TARA vNext.

Uses Hugging Face tokenizers for efficient BPE training/encoding.
"""

from src.tara_mind.core.tokenizer import FastBPETokenizer

__all__ = ["FastBPETokenizer"]
