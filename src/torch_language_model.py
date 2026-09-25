"""Accelerated tensor implementation of TARA's decoder language model.

The model keeps the original explicit Transformer design while supporting a
configurable depth, dropout, RMSNorm, and optional input/output embedding
weight tying for stronger experiments.
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
    def __init__(self, dimension, num_heads, dropout=0.0):
        super().__init__()
        if dimension <= 0 or num_heads <= 0:
            raise ValueError("dimension and num_heads must be positive")
        if dimension % num_heads:
            raise ValueError("dimension must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.dimension = dimension
        self.num_heads = num_heads
        self.head_dim = dimension // num_heads
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.output = nn.Linear(dimension, dimension)
        self.dropout = dropout

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
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, dimension)
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(self, dimension, ff_dimension, num_heads, dropout=0.0):
        super().__init__()
        if dimension <= 0 or ff_dimension <= 0:
            raise ValueError("dimension and ff_dimension must be positive")
        if num_heads <= 0 or dimension % num_heads:
            raise ValueError("dimension must be divisible by num_heads")
        self.norm1 = nn.RMSNorm(dimension)
        self.attention = CausalSelfAttention(dimension, num_heads, dropout=dropout)
        self.norm2 = nn.RMSNorm(dimension)
        self.ff = nn.Sequential(
            nn.Linear(dimension, ff_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class FastTinyLanguageModel(nn.Module):
    """Batch-first decoder Transformer with configurable depth and regularization."""

    def __init__(
        self,
        vocab_size,
        embedding_dim=32,
        ff_dim=64,
        num_heads=4,
        max_context=128,
        num_layers=1,
        dropout=0.0,
        tie_embeddings=False,
        seed=0,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if embedding_dim <= 0 or ff_dim <= 0 or num_heads <= 0:
            raise ValueError("embedding_dim, ff_dim, and num_heads must be positive")
        if max_context <= 0 or num_layers <= 0:
            raise ValueError("max_context and num_layers must be positive")
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.position = SinusoidalPositions(max_context, embedding_dim)
            self.transformer = nn.ModuleList(
                TransformerBlock(embedding_dim, ff_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            )
            self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=not tie_embeddings)
            if tie_embeddings:
                self.lm_head.weight = self.embedding.weight

        self.vocab_size = vocab_size
        self.max_context = max_context
        self.num_layers = num_layers
        self.dropout = dropout
        self.tie_embeddings = tie_embeddings

    def forward(self, token_ids):
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] == 0:
            raise ValueError("token_ids must contain at least one token")
        if token_ids.shape[1] > self.max_context:
            raise ValueError("sequence exceeds max_context")
        x = self.position(self.embedding(token_ids))
        for block in self.transformer:
            x = block(x)
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
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

    @torch.no_grad()
    def next_token(self, token_ids):
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] == 0:
            raise ValueError("token_ids must contain at least one token")
        logits = self(token_ids)[:, -1, :]
        return logits.argmax(dim=-1)

    @torch.no_grad()
    def sample_next_token(self, token_ids, temperature=1.0, top_k=None, rng=None):
        """Sample one next token for a single-token-sequence batch."""
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        context = torch.tensor([token_ids[-self.max_context:]], dtype=torch.long)
        logits = self(context)[0, -1, :] / temperature
        if top_k is not None:
            top_k = min(top_k, logits.numel())
            values, indices = torch.topk(logits, top_k)
            probs = torch.softmax(values, dim=-1)
            generator = None
            if rng is not None:
                seed = int(rng.random() * (2**63 - 1))
                generator = torch.Generator(device=probs.device)
                generator.manual_seed(seed)
            selected = torch.multinomial(probs, 1, generator=generator)
            return int(indices[selected].item())
        probs = torch.softmax(logits, dim=-1)
        generator = None
        if rng is not None:
            seed = int(rng.random() * (2**63 - 1))
            generator = torch.Generator(device=probs.device)
            generator.manual_seed(seed)
        return int(torch.multinomial(probs, 1, generator=generator).item())
