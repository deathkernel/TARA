"""Accelerated decoder-only Transformer used by TARA's vNext neural core.

Architecture: token embeddings -> RoPE causal attention -> RMSNorm ->
SwiGLU feed-forward blocks -> LM head.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_context, base=10000.0):
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be a positive even number")
        if max_context <= 0:
            raise ValueError("max_context must be positive")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_context, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, query, key):
        seq = query.shape[-2]
        cos = self.cos[:, :, :seq, :]
        sin = self.sin[:, :, :seq, :]
        def rotate(x):
            even, odd = x[..., 0::2], x[..., 1::2]
            return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
        return rotate(query), rotate(key)


class CausalSelfAttention(nn.Module):
    def __init__(self, dimension, num_heads, max_context, dropout=0.0):
        super().__init__()
        if dimension <= 0 or num_heads <= 0 or dimension % num_heads:
            raise ValueError("dimension must be positive and divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dimension // num_heads
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.output = nn.Linear(dimension, dimension)
        self.rope = RotaryEmbedding(self.head_dim, max_context)
        self.dropout = dropout

    def forward(self, x):
        batch, sequence, dimension = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, sequence, dimension)
        return self.output(y)


class SwiGLU(nn.Module):
    def __init__(self, dimension, ff_dimension, dropout=0.0):
        super().__init__()
        self.gate = nn.Linear(dimension, ff_dimension)
        self.up = nn.Linear(dimension, ff_dimension)
        self.down = nn.Linear(ff_dimension, dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dimension, ff_dimension, num_heads, max_context, dropout=0.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(dimension)
        self.attention = CausalSelfAttention(dimension, num_heads, max_context, dropout)
        self.norm2 = nn.RMSNorm(dimension)
        self.ff = SwiGLU(dimension, ff_dimension, dropout)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.ff(self.norm2(x))


class FastTinyLanguageModel(nn.Module):
    """Compatibility-preserving name for TARA's decoder Transformer."""

    def __init__(self, vocab_size, embedding_dim=32, ff_dim=64, num_heads=4,
                 max_context=128, num_layers=1, dropout=0.0,
                 tie_embeddings=False, seed=0):
        super().__init__()
        if vocab_size <= 0 or embedding_dim <= 0 or ff_dim <= 0 or num_heads <= 0:
            raise ValueError("model dimensions must be positive")
        if max_context <= 0 or num_layers <= 0:
            raise ValueError("max_context and num_layers must be positive")
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if (embedding_dim // num_heads) % 2:
            raise ValueError("head dimension must be even for RoPE")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.transformer = nn.ModuleList([
                TransformerBlock(embedding_dim, ff_dim, num_heads, max_context, dropout)
                for _ in range(num_layers)
            ])
            self.final_norm = nn.RMSNorm(embedding_dim)
            self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=not tie_embeddings)
            self._initialize_weights()
            if tie_embeddings:
                self.lm_head.weight = self.embedding.weight
        self.vocab_size = vocab_size
        self.max_context = max_context
        self.num_layers = num_layers
        self.dropout = dropout
        self.tie_embeddings = tie_embeddings

    def _initialize_weights(self):
        # Small GPT-style initialization keeps initial logits well-scaled.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, token_ids):
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] == 0:
            raise ValueError("token_ids must contain at least one token")
        if token_ids.shape[1] > self.max_context:
            raise ValueError("sequence exceeds max_context")
        x = self.embedding(token_ids)
        for block in self.transformer:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    def loss(self, inputs, targets):
        if inputs.ndim != 2 or targets.ndim != 2 or inputs.shape != targets.shape:
            raise ValueError("inputs and targets must have the same [batch, sequence] shape")
        if inputs.shape[1] == 0:
            raise ValueError("sequence must contain at least one token")
        if torch.any(targets < 0) or torch.any(targets >= self.vocab_size):
            raise ValueError("targets contain token ids outside the vocabulary")
        logits = self(inputs)
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

    @torch.no_grad()
    def next_token(self, token_ids):
        return self(token_ids)[:, -1, :].argmax(dim=-1)

    @torch.no_grad()
    def sample_next_token(self, token_ids, temperature=1.0, top_k=None, rng=None):
        """Sample one next token while preserving the legacy TARA brain API."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        logits = self(token_ids)[:, -1, :][0] / temperature
        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            top_k = min(int(top_k), logits.numel())
            values, indices = torch.topk(logits, top_k)
            probabilities = torch.softmax(values, dim=-1)
            choice = torch.multinomial(probabilities, 1, generator=rng)
            return int(indices[choice].item())
        probabilities = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probabilities, 1, generator=rng).item())
