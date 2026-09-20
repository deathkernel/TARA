# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a research-first neural-network project built from mathematical and implementation fundamentals. The goal is to understand the machinery behind modern neural networks before adding higher-level capabilities such as language modeling and voice interaction.

## Research-first rule

TARA follows:

**research → mathematical formulation → minimal experiment → own implementation → tests/benchmark → documentation**

Primary references include back-propagation work by Rumelhart, Hinton & Williams, Goodfellow, Bengio & Courville's *Deep Learning*, and Vaswani et al.'s *Attention Is All You Need*. Modern language-model ideas are introduced only after the underlying mechanisms are verified.

## Current milestone: tiny Transformer block

TARA now contains the main building pieces needed to study a small decoder-style Transformer path:

- Custom scalar reverse-mode automatic differentiation (`src/autograd.py`)
- Fully connected MLP components and MSE training
- Character/token embedding layer
- Numerically stable softmax
- Learned single-head Q/K/V causal self-attention (`src/attention.py`)
- Sinusoidal positional encoding (`src/positional.py`)
- Residual attention projection and position-wise feed-forward network (`src/transformer.py`)
- Deterministic unit tests for attention and Transformer invariants
- XOR convergence and numerical gradient-check experiments

The Transformer block intentionally uses **one attention head** and currently omits LayerNorm. These are deliberate simplifications for inspectability and CPU-friendly execution, not claims that the block reproduces a production GPT/Gemini architecture.

## Research notes

The original Transformer uses attention plus position-wise feed-forward layers, with residual connections and LayerNorm; decoder self-attention is causally masked so a position cannot use future tokens. TARA implements the causal attention mechanism, residual path, positional information, and feed-forward path in a much smaller scalar-autograd form. citeturn0search5

The standard scaled dot-product attention uses learned query, key and value projections and scales dot products by the square root of the key dimension. TARA's single-head implementation follows that core equation while keeping the dimensions tiny. citeturn0search8

Gemini 1.5 is a modern multimodal Transformer-family system with a very different scale and engineering envelope. It is used here as a research reference for understanding modern model direction, not as a blueprint to reproduce at home. citeturn0academia1

## Verification

The earlier scalar MLP milestone has a verified XOR experiment reaching approximately `5.27e-30` mean squared error with deterministic initialization. The new attention/Transformer tests have been added to the repository; they should be run locally before treating this milestone as fully verified.

## Project structure

```text
TARA/
├── src/
│   ├── autograd.py
│   ├── attention.py
│   ├── datasets.py
│   ├── embeddings.py
│   ├── gradcheck.py
│   ├── layers.py
│   ├── losses.py
│   ├── metrics.py
│   ├── mlp.py
│   ├── positional.py
│   ├── tokenizer.py
│   ├── training.py
│   └── transformer.py
├── tests/
│   ├── test_attention.py
│   ├── test_autograd.py
│   ├── test_datasets.py
│   ├── test_embeddings.py
│   ├── test_gradcheck.py
│   ├── test_metrics.py
│   ├── test_mlp.py
│   ├── test_tokenizer.py
│   ├── test_transformer.py
│   └── test_xor.py
├── experiments/
├── train_neuron.py
├── train_xor.py
├── requirements.txt
└── README.md
```

## Roadmap

1. ✅ Scalar autodiff
2. ✅ Backpropagation through an MLP
3. ✅ Verified XOR learning
4. ✅ Numerical gradient checking
5. ✅ Tokenization and embeddings
6. ✅ Single-head causal self-attention
7. ✅ Tiny Transformer block
8. ⬜ Tiny next-token language model
9. ⬜ Minimal local text generation/chat loop
10. ⬜ Research notes, final verification and cleanup
11. ⬜ Optional voice layer, kept separate from the neural-network core

TARA stays intentionally small: the objective is to understand and implement the core ideas ourselves, not to imitate the scale of frontier systems. The original Transformer paper establishes the architectural foundation, while later systems such as Gemini demonstrate how far Transformer-based models can be scaled and extended. citeturn0academia0turn0academia1
