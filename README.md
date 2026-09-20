# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a research-first neural-network project built from mathematical and implementation fundamentals. The goal is to understand the machinery behind modern neural networks before adding higher-level capabilities such as language modeling and voice interaction.

## Research-first rule

TARA is developed using this loop:

**research → mathematical formulation → minimal experiment → own implementation → tests/benchmark → documentation**

Primary references currently include Rumelhart, Hinton & Williams (1986) on back-propagation and Goodfellow, Bengio & Courville's *Deep Learning* textbook. Modern Transformer/language-model work will be introduced only after the core training machinery is verified.

## Current milestone: verified scalar training core

- Custom scalar reverse-mode automatic differentiation (`src/autograd.py`)
- Addition, multiplication, powers, division, ReLU and tanh derivatives
- Autograd-native fully connected neuron, layer and MLP (`src/mlp.py`)
- Tanh hidden layers with a linear output layer
- MSE loss and batch mean utilities (`src/losses.py`)
- Explicit SGD optimizer and training step (`src/training.py`)
- Centered finite-difference gradient checking (`src/gradcheck.py`)
- Unit tests for autodiff and MLP gradient correctness
- XOR convergence experiment with deterministic initialization

## Verification

The current XOR configuration trains a `2 → 3 → 3 → 1` MLP with tanh hidden layers. The verified local run reaches approximately `5.27e-30` mean squared error and predicts the four XOR targets to floating-point precision.

The gradient-check test compares every model parameter's autodiff gradient with a centered finite-difference estimate. This is an important correctness gate before adding more complex components.

## Project structure

```text
TARA/
├── src/
│   ├── autograd.py
│   ├── gradcheck.py
│   ├── layers.py          # legacy NumPy scaffold; not used by the MLP core yet
│   ├── losses.py
│   ├── mlp.py
│   └── training.py
├── tests/
│   ├── test_autograd.py
│   ├── test_gradcheck.py
│   └── test_xor.py
├── train_neuron.py
├── train_xor.py
└── README.md
```

## Roadmap

1. ✅ Scalar autodiff
2. ✅ Backpropagation through an MLP
3. ✅ Verified XOR learning
4. ✅ Numerical gradient checking
5. ⬜ Cleaner dataset/training abstractions and regression/classification experiments
6. ⬜ Tokenization and embeddings
7. ⬜ Self-attention
8. ⬜ Transformer block
9. ⬜ Small language model
10. ⬜ Instruction tuning / assistant behavior
11. ⬜ Voice layer (kept separate from the neural-network core)

TARA is intentionally built from small, inspectable components so every major capability can be tested before the next layer is introduced.
