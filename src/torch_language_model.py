"""Accelerated tensor implementation of TARA's tiny decoder LM.

This is a performance backend, not a replacement for TARA's scalar research
implementation. The scalar model remains useful for inspecting every
operation; this backend uses PyTorch tensors and automatic differentiation so
experiments can use larger batches and contexts.

Architecture:
    token embedding + sinusoidal positions
        -> pre-norm causal multi-head self-attention
        -> residual
        -> pre-norm GELU MLP
        -> residual
        -> vocabulary projection

PyTorch exposes scaled_dot_product_attention as a low-level Transformer
building block with optimized implementations; TARA uses the same primitive
while keeping the block structure explicit.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalPositions(nn.Module):
    def __init__(self, max_context, dimension):
        super().__init__()
        if max_context <= 0 or dimension <= 0:
            raise ValueError("max_context and dimension must be positive")

        positions = torch.arange(max_context, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dimension)
        )
        table = torch.zeros(max_context, dimension)
        table[:, 0::2] = torch.sin(positions * div_term)
        if dimension > 1:
            table[:, 1::2] = torch.cos(positions * div_term[: table[:, 1::2].shape[1]])

        self.register_buffer("table", table.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.table[:, : x.shape[1], :]


class CausalSelfAttention(nn.Module):
    def __init__(self, dimension, num_heads):
        super().__init__()
        if dimension <= 0 or num_heads <= 0:
            raise ValueError("dimension and num_heads must be positive")
        if dimension % num_heads:
            raise ValueError("dimension must be divisible by num_heads")

        self.dimension = dimension
        self.num_heads = num_heads
        self.head_dim = dimension // num_heads
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.output = nn.Linear(dimension, dimension)

    def forward(self, x):
        batch, sequence, dimension = x.shape

        qkv = self.qkv(x)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
        )

        attended = attended.transpose(1, 2).contiguous().view(
            batch, sequence, dimension
        )
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(self, dimension, ff_dimension, num_heads):
        super().__init__()
        if dimension <= 0 or ff_dimension <= 0:
            raise ValueError("dimension and ff_dimension must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if dimension % num_heads:
            raise ValueError("dimension must be divisible by num_heads")
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = CausalSelfAttention(dimension, num_heads)
        self.norm2 = nn.LayerNorm(dimension)
        self.ff = nn.Sequential(
            nn.Linear(dimension, ff_dimension),
            nn.GELU(),
            nn.Linear(ff_dimension, dimension),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class FastTinyLanguageModel(nn.Module):
    """Batch-first tensor implementation of the TARA decoder LM."""

    def __init__(
        self,
        vocab_size,
        embedding_dim=32,
        ff_dim=64,
        num_heads=4,
        max_context=128,
        seed=0,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if ff_dim <= 0:
            raise ValueError("ff_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if max_context <= 0:
            raise ValueError("max_context must be positive")
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.position = SinusoidalPositions(max_context, embedding_dim)
            self.transformer = TransformerBlock(
                embedding_dim,
                ff_dim,
                num_heads,
            )
            self.lm_head = nn.Linear(embedding_dim, vocab_size)

        self.vocab_size = vocab_size
        self.max_context = max_context

    def forward(self, token_ids):
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] == 0:
            raise ValueError("token_ids must contain at least one token")
        if token_ids.shape[1] > self.max_context:
            raise ValueError("sequence exceeds max_context")

        x = self.embedding(token_ids)
        x = self.position(x)
        x = self.transformer(x)
        return self.lm_head(x)

    def loss(self, inputs, targets):
        if inputs.ndim != 2 or targets.ndim != 2:
            raise ValueError("inputs and targets must have shape [batch, sequence]")
        if inputs.shape != targets.shape:
            raise ValueError("inputs and targets must have the same shape")
        if inputs.shape[1] == 0:
            raise ValueError("inputs and targets must contain at least one token")
        if torch.any(targets < 0) or torch.any(targets >= self.vocab_size):
            raise ValueError("targets contain token ids outside the vocabulary")

        logits = self(inputs)
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )

    @torch.no_grad()
    def next_token(self, token_ids):
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] == 0:
            raise ValueError("token_ids must contain at least one token")
        logits = self(token_ids)[:, -1, :]
        return logits.argmax(dim=-1)
