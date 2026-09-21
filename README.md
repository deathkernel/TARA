# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a research-first neural-network project built from mathematical and implementation fundamentals. The goal is a small, inspectable artificial reasoning system that can connect a neural language core with perception, memory, reasoning, planning and, only after the AI/model architecture is complete, controlled PC tools.

## Research-first rule

TARA follows:

**research → mathematical formulation → minimal experiment → own implementation → tests/benchmark → documentation**

Primary references include Rumelhart, Hinton & Williams on back-propagation, Goodfellow, Bengio & Courville's *Deep Learning*, Glorot & Bengio on initialization, Vaswani et al. on Transformers, Kingma & Ba on Adam, Loshchilov & Hutter on AdamW, and research on retrieval, deliberate reasoning, planning and agent loops.

TARA does **not** lock itself to one language, algorithm, or implementation style. The method is selected per problem: Python is used for the inspectable learning core and experiments, while Rust is already used for tokenizer work where a systems-oriented implementation is useful. Future components will be chosen by measured correctness, capability, resource cost, and research evidence rather than by forcing everything into one stack.

## Current architecture

TARA has a small autoregressive neural path built from its own scalar autodiff engine:

- Custom scalar reverse-mode automatic differentiation (`src/autograd.py`)
- MLP components and MSE training
- Deterministic character and BPE/subword tokenization
- Embedding table
- Numerically stable softmax/logsumexp and cross-entropy
- Learned Q/K/V causal self-attention
- Multi-head causal self-attention
- Sinusoidal positional encoding
- Pre-norm LayerNorm with learnable parameters
- Residual attention and position-wise feed-forward network
- GELU-style activation
- Configurable decoder-style Transformer stack
- Xavier/Glorot initialization
- AdamW adaptive optimization with decoupled weight decay
- Learning-rate scheduling and gradient clipping
- Tiny and PC-oriented small language-model profiles
- Checkpointing and deterministic resume support
- Greedy, temperature, top-k and top-p generation
- Language-model loss, perplexity and next-token accuracy benchmarks
- Training/validation diagnostics including predictive entropy and corpus statistics
- Bounded working memory and persistent long-term memory
- Deterministic retrieval, relevance scoring and forgetting
- Goal representation, task decomposition, planning, verification and re-planning
- File/document, system-state and screen-state perception representations
- Cognitive-state integration layer
- Agent loop with observation, persistent memory, verification and recovery
- Explicit permission, destructive-action confirmation and audit logging
- Controlled repository self-testing and structured pytest evidence

The language model is intentionally small and CPU-friendly. TARA is an educational/research implementation of the underlying mechanisms, not a reproduction of GPT, Gemini, or any frontier model.

## Cognitive layers

```text
Input / Perception
       ↓
Working + Long-Term Memory
       ↓
Neural Language Core
       ↓
Reasoning / Planning
       ↓
Agent Control Loop
       ↓
Verification / Recovery
       ↓
Evaluation + Safety
```

The PC-tool layer is intentionally **not** part of the current AI completion milestone. It will be added only after the complete AI/model architecture is developed, tested and evaluated.

## Memory

TARA separates:

- **Working memory** — bounded recent context.
- **Long-term memory** — persistent human-readable storage.
- **Retrieval** — deterministic lexical relevance scoring.
- **Memory dynamics** — access tracking, explicit updates and deterministic least-used forgetting.

The current retriever is deliberately transparent rather than a large vector system. It is a baseline for later learned retrieval.

## Reasoning

TARA's reasoning layer is deliberately explicit and deterministic. It provides:

- structured goals and success conditions
- task decomposition
- sequential plans
- step and goal verification
- failure-triggered re-planning

The design is informed by Tree of Thoughts, ReAct and reasoning-as-planning research, while remaining small enough to inspect and test.

## Perception and integration

Perception converts externally observable inputs into bounded structured representations without performing actions:

- text document state
- normalized system facts
- deterministic screen-element summaries

The integration layer maintains goal, plan, observations and results. The agent layer can persist observations, retrieve relevant memories, select planned tasks, accept externally supplied results, verify them, advance successful steps and recover through explicit re-planning.

## Safety

Safety is a separate boundary around future tools. The current foundation uses:

- default-deny permissions
- explicit permission changes
- confirmation gates for designated destructive actions
- immutable authorization decisions
- an audit log

No PC action is executed by these modules.

## Evaluation and self-testing

TARA keeps measured evidence separate from implementation. Evaluation supports:

- metric acceptance ranges
- baseline comparisons
- finite-difference gradient validation
- failure classification
- JSON-serializable reports
- corpus statistics and held-out evaluation

The self-test runner executes only the repository's fixed `python -m pytest -q` suite and returns command, exit status and captured output. It does not execute arbitrary commands supplied by a model.

## Run locally

From the repository root:

```bash
python -m pytest -q
python train_xor.py
python train_tiny_lm.py
python experiments/scaled_lm_diagnostics.py
python experiments/train_scaled_lm.py
python experiments/heldout_generalization.py
```

## Run automatically on GitHub

TARA includes GitHub Actions workflows under `.github/workflows/`. They install Python and run the test suite on GitHub-hosted runners. The workflow with `workflow_dispatch` can also be started manually from the **Actions** tab.

## Project structure

```text
TARA/
├── .github/workflows/
├── experiments/
│   ├── heldout_generalization.py
│   ├── scaled_lm_diagnostics.py
│   └── train_scaled_lm.py
├── src/
│   ├── activations.py
│   ├── agent.py
│   ├── autograd.py
│   ├── checkpoint.py
│   ├── dataset_registry.py
│   ├── datasets.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── generation.py
│   ├── gradcheck.py
│   ├── gradient_clipping.py
│   ├── initialization.py
│   ├── integration.py
│   ├── language_benchmarks.py
│   ├── language_dataset.py
│   ├── language_model.py
│   ├── layers.py
│   ├── losses.py
│   ├── memory.py
│   ├── metrics.py
│   ├── mlp.py
│   ├── optimizers.py
│   ├── perception.py
│   ├── positional.py
│   ├── reasoning.py
│   ├── safety.py
│   ├── schedulers.py
│   ├── self_test.py
│   ├── text_dataset.py
│   ├── tokenizer.py
│   └── transformer.py
├── tests/
├── train_neuron.py
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
12. ✅ Optimizer
13. ✅ Learning-rate scheduling
14. ✅ Gradient clipping

### Phase 2 — Language Intelligence

15. ✅ BPE/subword tokenizer
16. ✅ Local corpus pipeline
17. ✅ Batching and validation
18. ✅ Training metrics and checkpoints
19. ✅ Generation controls and evaluation

### Phase 3 — Memory

20. ✅ Working memory
21. ✅ Long-term memory
22. ✅ Retrieval and relevance scoring
23. ✅ Memory update / forgetting

### Phase 4 — Reasoning

24. ✅ Goal representation
25. ✅ Task decomposition
26. ✅ Planning
27. ✅ Verification
28. ✅ Error recovery and re-planning

### Phase 5 — Perception

29. ✅ File/document perception
30. ✅ System-state perception
31. ✅ Screen-state representation

### Phase 6 — AI Cognitive Integration

32. ✅ Cognitive state container
33. ✅ Perception/memory/reasoning integration boundary
34. ✅ Structured state snapshots

### Phase 7 — Agent / Brain Loop

35. ✅ Observe
36. ✅ Remember / retrieve
37. ✅ Select next planned task
38. ✅ Verify result and advance
39. ✅ Failure recovery and re-planning

### Phase 8 — Safety Foundation

40. ✅ Default-deny permissions
41. ✅ Destructive-action confirmation
42. ✅ Action-decision audit logging

### Phase 9 — AI Self-testing and Evaluation

43. ✅ Automatic repository test execution
44. ✅ Structured failure evidence
45. ✅ Regression/metric comparison primitives
46. ✅ Mathematical gradient evidence
47. ✅ Deterministic self-test coverage

### Phase 10 — Scientific Scaling

48. ⬜ Train and benchmark the PC-oriented scaled model
49. ⬜ Compare capacity against the tiny baseline
50. ⬜ Evaluate data/capacity/compute trade-offs
51. ⬜ Improve training data and language evaluation
52. ⬜ Compare training optimizers under a matched experimental budget

### Post-AI Phase — PC Tools

Only after the AI/model architecture is complete and evaluated:

53. ⬜ File tools
54. ⬜ Terminal tools
55. ⬜ Application/browser tools
56. ⬜ Controlled keyboard/mouse interaction
57. ⬜ Tool selection and execution verification
58. ⬜ Emergency stop / rollback where possible

TARA stays intentionally small. The objective is to understand and implement the core mechanisms ourselves, validate them mathematically and empirically, then connect them into a useful reasoning-and-tool system rather than imitate frontier-model scale.
