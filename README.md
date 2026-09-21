# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a research-first neural-network project built from mathematical and implementation fundamentals. The goal is to understand the machinery behind modern neural networks before adding higher-level capabilities such as language modeling and voice interaction.

## Research-first rule

TARA follows:

**research → mathematical formulation → minimal experiment → own implementation → tests/benchmark → documentation**

Primary references include back-propagation work by Rumelhart, Hinton & Williams, Goodfellow, Bengio & Courville's *Deep Learning*, and Vaswani et al.'s *Attention Is All You Need*. Modern language-model ideas are introduced only after the underlying mechanisms are verified.

## Current milestone: tiny next-token language model

TARA now has a complete minimal path from characters to autoregressive next-token prediction:

- Custom scalar reverse-mode automatic differentiation (`src/autograd.py`)
- Fully connected MLP components and MSE training
- Deterministic character tokenizer and embedding table
- Numerically stable softmax
- Learned single-head Q/K/V causal self-attention (`src/attention.py`)
- Sinusoidal positional encoding (`src/positional.py`)
- Residual attention projection and position-wise feed-forward network (`src/transformer.py`)
- Tiny character-level language model with a vocabulary projection (`src/language_model.py`)
- Cross-entropy next-token loss and greedy next-token prediction
- Unit tests for attention, Transformer structure, language-model shapes and backpropagation

The language model is intentionally tiny and CPU-friendly. It is an educational implementation of the autoregressive mechanism, not a reproduction of GPT, Gemini, or any frontier model.

## Research notes

The Transformer architecture combines attention with position-wise feed-forward layers, residual connections and normalization; decoder self-attention is causally masked so a position cannot use future tokens. TARA keeps the causal dependency while reducing the architecture to one head and one small block for inspectability.

Causal language modeling trains the model to predict the next token from the available left context. A sequence is shifted by one position so the input at each location is used to predict its following target.

Scaled dot-product attention uses learned query, key and value projections and scales the dot product by the square root of the key dimension. TARA follows this core mechanism with tiny scalar-autograd matrices.

## Run locally

From the repository root:

```bash
python -m pytest -q
python train_xor.py
python train_tiny_lm.py
```

`train_xor.py` runs the small XOR learning experiment. `train_tiny_lm.py` trains the tiny character-level language model on its built-in local corpus and then generates text.

## Run automatically on GitHub

TARA includes a GitHub Actions workflow at `.github/workflows/python-tests.yml`. GitHub-hosted runners can install Python and run the same test commands used locally.

The workflow runs automatically when code is pushed to `main` or when a pull request targets `main`. It also supports a manual run:

1. Open the TARA repository on GitHub.
2. Open the **Actions** tab.
3. Select **TARA Tests**.
4. Click **Run workflow**.
5. Select `main` and click **Run workflow**.
6. Open the new run to see the test and XOR experiment output.

The workflow executes:

```bash
python -m pytest -q
python train_xor.py
```

This gives TARA a basic continuous-integration check on GitHub while keeping the project small.

## Verification status

The scalar MLP milestone has a verified XOR experiment reaching approximately `5.27e-30` mean squared error with deterministic initialization. The attention, Transformer and language-model tests are committed; GitHub Actions now provides an automated environment for running the test suite and XOR experiment.

## Project structure

```text
TARA/
├── .github/
│   └── workflows/
│       └── python-tests.yml
├── src/
│   ├── attention.py
│   ├── autograd.py
│   ├── datasets.py
│   ├── embeddings.py
│   ├── gradcheck.py
│   ├── language_model.py
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
│   ├── test_language_model.py
│   ├── test_metrics.py
│   ├── test_mlp.py
│   ├── test_tokenizer.py
│   ├── test_transformer.py
│   └── test_xor.py
├── experiments/
├── train_neuron.py
├── train_xor.py
├── train_tiny_lm.py
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
8. ✅ Tiny next-token language-model core
9. ⬜ Tiny local training experiment and text generation
10. ⬜ Final research notes, verification and cleanup
11. ⬜ Optional voice layer, kept separate from the neural-network core

TARA stays intentionally small: the objective is to understand and implement the core ideas ourselves, not to imitate the scale of frontier systems.
