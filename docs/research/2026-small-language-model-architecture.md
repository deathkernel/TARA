# Research note: TARA's small decoder language model

**Status:** experimental architecture, not the default runtime model.  
**Implementation:** `src/research_language_model.py`  
**Tests:** `tests/test_research_language_model.py`

## Design decisions grounded in papers

1. **Rotary position embeddings (RoPE).** Applies position-dependent rotations to query/key pairs so attention can encode relative position. The implementation computes the rotary frequencies from the active sequence length and device instead of storing a fixed positional table. See [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://consensus.app/papers/roformer-enhanced-transformer-with-rotary-position-su-ahmed/35d8d87ad3ec52c49713f6ed2656104b/?utm_source=chatgpt).
2. **Grouped-query attention (GQA).** Keeps multiple query heads but shares each key/value head across a group, reducing key/value projection width and potentially the decode-time KV footprint. This reference implementation expands K/V heads for PyTorch's standard scaled-dot-product attention, so it validates the architecture but does not yet implement a persistent KV cache. See [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://consensus.app/papers/gqa-training-generalized-multiquery-transformer-models-ainslie-lee-thorp/169563f020cf54aeacc517b9abeab2f8/?utm_source=chatgpt).
3. **SwiGLU feed-forward blocks.** Uses `SiLU(gate(x)) * up(x)` followed by a down projection, reflecting the gated feed-forward variants studied in [GLU Variants Improve Transformer](https://consensus.app/papers/glu-variants-improve-transformer-shazeer/a61336c71bd8535aa0f6696f4a7a7bda/?utm_source=chatgpt).
4. **RMSNorm and pre-norm residual blocks.** Keeps normalization before attention and feed-forward sublayers, consistent with modern small decoder implementations. The architecture is deliberately small and configurable so TARA can compare it against its existing baseline.

## What the papers do—and do not—justify

The cited work motivates testing these components; it does **not** prove they improve TARA on its current data, tokenizer, or hardware. The GQA projection width is reduced in this implementation, but K/V tensors are repeated before attention and no cache is implemented, so end-to-end speedups must not be claimed. RoPE and SwiGLU are not assumed to win without controlled evaluation.

## Reproducible validation plan

- Keep `FastTinyLanguageModel` as the untouched baseline.
- Use the same tokenizer, train/validation split, seed, context length, batch size, optimizer, token budget, and evaluation batches for both architectures.
- Record validation negative log-likelihood, parameter count, training tokens/second, peak memory when available, and generation latency.
- Run at least three seeds before making a default-architecture decision.
- Never use the validation split for optimizer/model selection and then report it as a final held-out test set.

## Research references

- Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv:2104.09864, 6,660 citations reported by Consensus. [Paper](https://consensus.app/papers/roformer-enhanced-transformer-with-rotary-position-su-ahmed/35d8d87ad3ec52c49713f6ed2656104b/?utm_source=chatgpt)
- Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, arXiv:2305.13245, 1,847 citations reported by Consensus. [Paper](https://consensus.app/papers/gqa-training-generalized-multiquery-transformer-models-ainslie-lee-thorp/169563f020cf54aeacc517b9abeab2f8/?utm_source=chatgpt)
- Shazeer (2020), *GLU Variants Improve Transformer*, arXiv:2002.05202, 2,298 citations reported by Consensus. [Paper](https://consensus.app/papers/glu-variants-improve-transformer-shazeer/a61336c71bd8535aa0f6696f4a7a7bda/?utm_source=chatgpt)

Citation counts are the metadata returned by Consensus at research time and will change. The sources are papers, not proof that this implementation has improved benchmark scores.
