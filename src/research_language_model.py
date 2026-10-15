"""Research-oriented decoder-only language model for TARA experiments.

This model is intentionally separate from FastTinyLanguageModel so research
changes can be benchmarked without silently changing the existing baseline.
It combines rotary position embeddings, grouped-query attention, RMSNorm, and
SwiGLU in a small, configurable PyTorch implementation.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent feature pairs by 90 degrees for RoPE."""
    if x.shape[-1] % 2:
        raise ValueError("the last dimension must be even for rotary embeddings")
    paired = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """Apply RoPE dynamically for the sequence length and device in the input."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be a positive even number")
        if base <= 1.0:
            raise ValueError("base must be greater than 1")
        inverse_frequencies = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inverse_frequencies", inverse_frequencies, persistent=False)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = query.shape[-2]
        positions = torch.arange(
            sequence_length,
            device=query.device,
            dtype=self.inverse_frequencies.dtype,
        )
        frequencies = torch.outer(positions, self.inverse_frequencies.to(query.device))
        angles = frequencies.repeat_interleave(2, dim=-1)
        # The repeated halves correspond to adjacent-pair rotation after rotate_half.
        cos = angles.cos()[None, None, :, :].to(dtype=query.dtype)
        sin = angles.sin()[None, None, :, :].to(dtype=query.dtype)
        return (
            query * cos + rotate_half(query) * sin,
            key * cos + rotate_half(key) * sin,
        )


class GroupedQueryAttention(nn.Module):
    """Causal attention with fewer key/value heads than query heads."""

    def __init__(
        self,
        dimension: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dimension <= 0 or num_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("dimension and attention head counts must be positive")
        if dimension % num_heads:
            raise ValueError("dimension must be divisible by num_heads")
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if (dimension // num_heads) % 2:
            raise ValueError("head dimension must be even for rotary embeddings")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.dimension = dimension
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dimension // num_heads
        self.dropout = dropout
        self.query = nn.Linear(dimension, num_heads * self.head_dim, bias=False)
        self.key = nn.Linear(dimension, num_kv_heads * self.head_dim, bias=False)
        self.value = nn.Linear(dimension, num_kv_heads * self.head_dim, bias=False)
        self.output = nn.Linear(dimension, dimension, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = x.shape
        query = self.query(x).view(
            batch, sequence, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.key(x).view(
            batch, sequence, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value = self.value(x).view(
            batch, sequence, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        query, key = self.rotary(query, key)
        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch, sequence, self.dimension
        )
        return self.output(attended)


class SwiGLU(nn.Module):
    """Gated feed-forward layer using SiLU(gate) multiplied by an up projection."""

    def __init__(self, dimension: int, ff_dimension: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dimension <= 0 or ff_dimension <= 0:
            raise ValueError("dimension and ff_dimension must be positive")
        self.gate = nn.Linear(dimension, ff_dimension, bias=False)
        self.up = nn.Linear(dimension, ff_dimension, bias=False)
        self.down = nn.Linear(ff_dimension, dimension, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class ResearchTransformerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        ff_dimension: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(dimension)
        self.attention = GroupedQueryAttention(
            dimension, num_heads, num_kv_heads, dropout=dropout
        )
        self.ffn_norm = nn.RMSNorm(dimension)
        self.ffn = SwiGLU(dimension, ff_dimension, dropout=dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_dropout(self.attention(self.attention_norm(x)))
        return x + self.residual_dropout(self.ffn(self.ffn_norm(x)))


class ResearchTinyLanguageModel(nn.Module):
    """Small decoder-only LM for controlled, research-driven experiments.

    The caller owns tokenization, data splits, optimizer selection, and
    checkpoint policy. This class does not download data or load pretrained
    weights; parameters are initialized from scratch.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        ff_dim: int = 128,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        max_context: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        tie_embeddings: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        if embedding_dim <= 0 or ff_dim <= 0:
            raise ValueError("embedding_dim and ff_dim must be positive")
        if num_layers <= 0 or max_context <= 0:
            raise ValueError("num_layers and max_context must be positive")
        if num_heads <= 0 or embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if num_kv_heads <= 0 or num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if (embedding_dim // num_heads) % 2:
            raise ValueError("head dimension must be even for rotary embeddings")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        # Keep construction deterministic without consuming the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.blocks = nn.ModuleList(
                ResearchTransformerBlock(
                    embedding_dim, ff_dim, num_heads, num_kv_heads, dropout
                )
                for _ in range(num_layers)
            )
            self.final_norm = nn.RMSNorm(embedding_dim)
            self.lm_head = nn.Linear(
                embedding_dim, vocab_size, bias=not tie_embeddings
            )
            if tie_embeddings:
                self.lm_head.weight = self.embedding.weight

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_context = max_context
        self.num_layers = num_layers
        self.dropout = dropout
        self.tie_embeddings = tie_embeddings

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] == 0:
            raise ValueError("token_ids must contain at least one token")
        if token_ids.shape[1] > self.max_context:
            raise ValueError("sequence exceeds max_context")
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or targets.ndim != 2:
            raise ValueError("inputs and targets must have shape [batch, sequence]")
        if inputs.shape != targets.shape:
            raise ValueError("inputs and targets must have the same shape")
        if inputs.shape[1] == 0:
            raise ValueError("inputs and targets must not be empty")
        if torch.any(targets < 0) or torch.any(targets >= self.vocab_size):
            raise ValueError("targets contain token ids outside the vocabulary")
        logits = self(inputs)
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample tokens; useful for smoke tests and local demos."""
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("token_ids must be a non-empty [batch, sequence] tensor")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")

        output = token_ids
        for _ in range(max_new_tokens):
            context = output[:, -self.max_context :]
            logits = self(context)[:, -1, :] / temperature
            if top_k is not None:
                values, indices = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
                probabilities = torch.softmax(values, dim=-1)
                sampled = indices.gather(
                    -1, torch.multinomial(probabilities, 1, generator=generator)
                )
            else:
                probabilities = torch.softmax(logits, dim=-1)
                sampled = torch.multinomial(probabilities, 1, generator=generator)
            output = torch.cat((output, sampled), dim=1)
        return output


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return a transparent parameter count for benchmark reports."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )
