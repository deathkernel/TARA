# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a research-first neural-network project built from mathematical and implementation fundamentals. The long-term goal is a small, inspectable artificial reasoning system that can eventually connect perception, memory, reasoning, planning and controlled PC tools while remaining practical on a normal PC.

## Research-first rule

TARA follows:

**research → mathematical formulation → minimal experiment → own implementation → tests/benchmark → documentation**

Primary references include back-propagation work by Rumelhart, Hinton & Williams, Goodfellow, Bengio & Courville's *Deep Learning*, Glorot & Bengio's initialization work, and Vaswani et al.'s *Attention Is All You Need*. Modern language-model ideas are introduced only after the underlying mechanisms are verified.

## Current architecture

TARA now has a small autoregressive neural path built from its own scalar autodiff engine:

- Custom scalar reverse-mode automatic differentiation (`src/autograd.py`)
- Fully connected MLP components and MSE training
- Deterministic character tokenizer and embedding table
- Numerically stable softmax and cross-entropy
- Learned Q/K/V causal self-attention
- Multi-head causal self-attention
- Sinusoidal positional encoding
- Pre-norm LayerNorm with learnable gamma and beta
- Residual attention and position-wise feed-forward network
- GELU-style activation
- Configurable stack of decoder-style Transformer blocks
- Xavier/Glorot initialization for the MLP and Transformer linear projections
- Tiny character-level autoregressive language model
- Greedy and temperature/top-k next-token sampling
- Automated tests for shapes, causality, gradients, normalization, initialization and language-model behavior

The language model is intentionally tiny and CPU-friendly. It is an educational implementation of the underlying mechanisms, not a reproduction of GPT, Gemini, or any frontier model.

## Parameter initialization

TARA uses Xavier/Glorot uniform initialization for its small tanh/GELU-style linear projections. The bound is:

`limit = sqrt(6 / (fan_in + fan_out))`

The implementation lives in `src/initialization.py` so initialization is explicit, deterministic when given a seeded random generator, and independently testable.

## Research notes

The Transformer architecture combines attention with position-wise feed-forward layers, residual connections and normalization; decoder self-attention is causally masked so a position cannot use future tokens. TARA keeps these core dependencies while reducing the model to a small, inspectable implementation.

Xavier/Glorot initialization was introduced to keep signal scales better behaved through layers during training. He initialization is a related activation-specific method designed particularly around rectifier nonlinearities; TARA currently uses Xavier/Glorot because its main small projections use tanh/GELU-style nonlinearities.

Causal language modeling trains the model to predict the next token from the available left context. A sequence is shifted by one position so the input at each location is used to predict its following target.

## Run locally

From the repository root:

```bash
python -m pytest -q
python train_xor.py
python train_tiny_lm.py
```

`train_xor.py` runs the small XOR learning experiment. `train_tiny_lm.py` trains the tiny character-level language model on its built-in local corpus and then generates text.

## Run automatically on GitHub

TARA includes GitHub Actions workflows under `.github/workflows/`. The workflows install Python and run the test suite on GitHub-hosted runners.

The test workflow runs automatically on pushes and pull requests. The workflow with `workflow_dispatch` can also be started manually from the **Actions** tab using **Run workflow**.

## Verification status

The scalar MLP milestone has a verified XOR experiment reaching approximately `5.27e-30` mean squared error with deterministic initialization. Attention, Transformer, LayerNorm, Transformer-stack, initialization and language-model tests are committed. GitHub Actions is used as an automated verification environment.

## Long-term direction

TARA's long-term architecture is intentionally staged:

```text
Perception
    ↓
Memory
    ↓
Neural Core
    ↓
Reasoning / Planning
    ↓
Tools
    ↓
Controlled PC interaction
    ↓
Observation / feedback
    └──────────────→ Memory / Reasoning
```

The PC-automation layer will be added only after the neural, memory, reasoning and tool interfaces are well-defined and tested. Destructive or irreversible actions will require explicit safety controls.

## Project structure

```text
TARA/
├── .github/
│   └── workflows/
│       ├── python-tests.yml
│       └── tests.yml
├── src/
│   ├── attention.py
│   ├── autograd.py
│   ├── datasets.py
│   ├── embeddings.py
│   ├── gradcheck.py
│   ├── initialization.py
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
│   ├── test_initialization.py
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

### Phase 1 — Neural Foundation

1. ✅ Scalar autodiff
2. ✅ Backpropagation through an MLP
3. ✅ Verified XOR learning
4. ✅ Numerical gradient checking
5. ✅ Tokenization and embeddings
6. ✅ Causal self-attention
7. ✅ LayerNorm
8. ✅ Multi-head attention
9. ✅ Transformer block
10. ✅ Configurable Transformer stack
11. ✅ Research-based parameter initialization
12. ⬜ Better optimizer
13. ⬜ Learning-rate scheduling
14. ⬜ Gradient clipping

### Phase 2 — Language Intelligence

15. ⬜ Better tokenizer / subword tokenizer
16. ⬜ Better local corpus pipeline
17. ⬜ Batching and validation
18. ⬜ Training metrics and checkpoints
19. ⬜ Generation controls and evaluation

### Phase 3 — Memory

20. ⬜ Working memory
21. ⬜ Long-term memory
22. ⬜ Retrieval and relevance scoring
23. ⬜ Memory update / forgetting

### Phase 4 — Reasoning

24. ⬜ Goal representation
25. ⬜ Task decomposition
26. ⬜ Planning
27. ⬜ Verification
28. ⬜ Error recovery and re-planning

### Phase 5 — Perception

29. ⬜ File/document perception
30. ⬜ System-state perception
31. ⬜ Screen-state representation

### Phase 6 — PC Tools

32. ⬜ File tools
33. ⬜ Terminal tools
34. ⬜ Application tools
35. ⬜ Browser tools
36. ⬜ Controlled keyboard/mouse tools

### Phase 7 — Agent Loop

37. ⬜ Observe → Remember → Reason → Plan → Act → Observe
38. ⬜ Tool selection
39. ⬜ Persistent task state

### Phase 8 — Safety

40. ⬜ Permissions
41. ⬜ Destructive-action confirmation
42. ⬜ Action logging
43. ⬜ Emergency stop / rollback where possible

### Phase 9 — Self-testing

44. ⬜ Automatic test execution
45. ⬜ Failure analysis
46. ⬜ Regression testing
47. ⬜ Action verification

TARA stays intentionally small: the objective is to understand and implement the core ideas ourselves, then gradually connect them into a useful reasoning-and-tool system rather than imitating frontier-model scale.
