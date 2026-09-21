"""Fast, non-autograd inference for TARA Transformer language models.

Training intentionally uses scalar ``Value`` objects so every derivative is
inspectable. Inference does not need a computation graph, so this module uses
NumPy arrays while reading the same learned scalar parameters. Keeping the
path separate prevents evaluation and generation from allocating enormous
training graphs on the CPU.
"""

import math

import numpy as np

from src.positional import add_sinusoidal_position


def _matrix(linear):
    return np.asarray([[value.data for value in row] for row in linear.w], dtype=float)


def _vector(values):
    return np.asarray([value.data for value in values], dtype=float)


def _linear(linear, x):
    return _matrix(linear) @ x + _vector(linear.b)


def _layer_norm(layer, x):
    mean = float(np.mean(x))
    centered = x - mean
    variance = float(np.mean(centered * centered))
    return _vector(layer.gamma) * centered / math.sqrt(variance + layer.eps) + _vector(layer.beta)


def _attention_head(head, sequence):
    q = sequence @ _matrix_from_rows(head.wq).T + _vector(head.bq)
    k = sequence @ _matrix_from_rows(head.wk).T + _vector(head.bk)
    v = sequence @ _matrix_from_rows(head.wv).T + _vector(head.bv)
    scale = math.sqrt(head.head_dim)
    outputs = np.empty_like(v)
    for position in range(len(sequence)):
        scores = (q[position] @ k[: position + 1].T) / scale
        maximum = float(np.max(scores))
        weights = np.exp(scores - maximum)
        weights /= float(np.sum(weights))
        outputs[position] = weights @ v[: position + 1]
    return outputs


def _matrix_from_rows(rows):
    return np.asarray([[value.data for value in row] for row in rows], dtype=float)


def _transformer_block(block, sequence):
    normalized = np.asarray([_layer_norm(block.norm1, token) for token in sequence])
    attended = [
        _attention_head(head, normalized)
        for head in block.attention.heads
    ]
    combined = np.concatenate(attended, axis=1)
    projected = combined @ _matrix(block.attention.output).T + _vector(block.attention.output.b)
    after_attention = sequence + projected

    output = np.empty_like(after_attention)
    for position, residual in enumerate(after_attention):
        normalized = _layer_norm(block.norm2, residual)
        hidden = normalized @ _matrix(block.ff1).T + _vector(block.ff1.b)
        hidden = 0.5 * hidden * (
            1.0 + np.tanh(
                math.sqrt(2.0 / math.pi)
                * (hidden + 0.044715 * hidden ** 3)
            )
        )
        update = hidden @ _matrix(block.ff2).T + _vector(block.ff2.b)
        output[position] = residual + update
    return output


def forward_numeric(model, token_ids):
    """Return model logits as plain floats without constructing an autograd graph."""
    if not token_ids:
        raise ValueError("token_ids must not be empty")

    vectors = np.asarray(
        [[value.data for value in model.embedding.forward(token_id)] for token_id in token_ids],
        dtype=float,
    )

    positions = np.arange(len(vectors), dtype=float)[:, None]
    dimensions = np.arange(model.embedding.embedding_dim, dtype=float)[None, :]
    angle_rates = 1.0 / (10000.0 ** (2.0 * np.floor(dimensions / 2.0) / model.embedding.embedding_dim))
    angles = positions * angle_rates
    encoded = vectors.copy()
    encoded[:, 0::2] += np.sin(angles[:, 0::2])
    encoded[:, 1::2] += np.cos(angles[:, 1::2])

    hidden = encoded
    for block in model.transformer.blocks:
        hidden = _transformer_block(block, hidden)

    head_matrix = _matrix(model.lm_head)
    logits = hidden @ head_matrix.T + _vector(model.lm_head.b)
    if not np.all(np.isfinite(logits)):
        raise ValueError("numeric inference produced non-finite logits")
    return logits.tolist()
