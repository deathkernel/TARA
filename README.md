# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a research-first neural-network project built from mathematical and implementation fundamentals. The goal is a small, inspectable artificial reasoning system that can connect a neural language core with perception, memory, reasoning, planning and, only after the AI/model architecture is complete, controlled PC tools.

## Research-first rule

TARA follows:

**research → mathematical formulation → minimal experiment → own implementation → tests/benchmark → documentation**

Primary references include Rumelhart, Hinton & Williams on back-propagation, Goodfellow, Bengio & Courville's *Deep Learning*, Glorot & Bengio on initialization, Vaswani et al. on Transformers, Kingma & Ba on Adam, Loshchilov & Hutter on AdamW, and research on retrieval, deliberate reasoning, planning and agent loops.

TARA does **not** lock itself to one language, algorithm, or implementation style. The method is selected per problem: Python is used for the inspectable learning core and experiments, while Rust is already used for tokenizer work where a systems-oriented implementation is useful. Future components will be chosen by measured correctness, capability, resource cost, and research evidence rather than by forcing everything into one stack.

## Current architecture

```text
Input / Perception
       ↓
Working Memory → Long-Term Memory
       ↓
Neural Language Core
       ↓
Reasoning → Planning
       ↓
Action / Tools
       ↓
Observation
       ↓
Verification → Reflection
       ↓
Learning
       ↓
Hypothesis → Experiment
       ↓
Knowledge Graph ↔ Research
       ↓
Multimodal Input
       ↓
Autonomous Orchestration
       ↺ feedback to perception
```

## Complete basic brain baseline

`src/basic_brain.py` now contains one coherent, dependency-light baseline for the complete cognitive architecture. It is intentionally simple so every layer is explicit and can later be upgraded independently without replacing the overall brain boundary.

The baseline includes:

- text, file, screen, audio and image perception representations
- bounded working memory
- deterministic long-term memory and retrieval
- goal and symbolic reasoning representations
- task planning and plan progression
- explicit allow-listed tool registry
- action results and failure capture
- deterministic result verification
- reflection and recovery lessons
- verified-only learning records
- hypothesis generation
- experiment execution through an injected test function
- knowledge-graph edges and retrieval
- an injected research-provider boundary with no implicit network access
- multimodal input packing
- `BasicTARABrain.run_cycle()` connecting the layers into one basic observe → reason → plan → act → verify → reflect → learn loop
- serializable high-level brain snapshots

This is a **basic architecture skeleton, not a claim of human-level intelligence**. The existing Transformer/language-model, memory, reasoning, agent, verification, polyglot algorithm-discovery and scientific-evaluation implementations remain the places where capability is developed. The basic brain gives them a single conceptual architecture to grow into.

Example:

```python
from src.basic_brain import BasicTARABrain, ToolRegistry

registry = ToolRegistry()
registry.register("add", lambda a, b: a + b)
brain = BasicTARABrain(tool_registry=registry)

result = brain.run_cycle(
    "calculate 2 + 3",
    goal="calculate the result",
    subtasks=["parse input", "calculate", "verify"],
    tool="add",
    tool_kwargs={"a": 2, "b": 3},
    expected=5,
    verify=True,
)

assert result.verification.passed
assert result.learned
```

The tool layer is default-deny: the brain cannot execute an unregistered command. The research boundary similarly requires the host to inject a provider. Arbitrary model-generated programs are not executed through this baseline.

## Completed AI/model layers

- Custom scalar reverse-mode automatic differentiation
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
- Train/validation and separate held-out evaluation
- Bounded working memory and persistent long-term memory
- Deterministic retrieval, relevance scoring and forgetting
- Goal representation, task decomposition, planning, verification and re-planning
- File/document, system-state and screen-state perception representations
- Cognitive-state integration layer
- Agent loop with observation, persistent memory, verification and recovery
- Explicit permission, destructive-action confirmation and audit logging
- Scientific scaling experiments with matched training budgets
- Integrated `TARABrain` boundary connecting language, perception, memory, reasoning and agent control without executing external actions
- Polyglot candidate generation, compilation, execution and benchmarking
- Candidate archive, structural similarity and verified-knowledge extraction
- Basic complete cognitive architecture in `src/basic_brain.py`

The language model is intentionally small and CPU-friendly. TARA is an educational/research implementation of the underlying mechanisms, not a reproduction of GPT, Gemini, or any frontier model.

## Scientific scaling

Phase 10 now has a matched-budget experiment comparing the tiny and PC-oriented scaled profiles. The protocol keeps corpus construction, tokenizer training, seed, batch size, context length, optimizer family, learning-rate schedule, gradient clipping and update count aligned. It records parameter count, token-budget proxy, train/validation loss, held-out loss/accuracy and gradient statistics.

Run it with:

```bash
python experiments/scientific_scaling.py
```

The experiment reports measurements; it does not declare a universal "better" profile. Capacity, data and compute are evaluated together.

## Integrated brain

`src/brain.py` provides the final pre-tool cognitive boundary. `TARABrain` can observe, remember/retrieve, set goals, create plans, generate language, verify externally supplied results and recover through re-planning. It deliberately does **not** execute shell commands, browser actions, keyboard/mouse input or other PC actions.

`src/basic_brain.py` complements this boundary with a complete basic layer map, including hypotheses, experiments, knowledge graph, research and multimodal interfaces. It is the simple architectural baseline that future advanced modules can replace incrementally.

## Safety and future tools

Safety remains a separate boundary around future tools. The current foundation uses:

- default-deny permissions
- explicit permission changes
- confirmation gates for designated destructive actions
- immutable authorization decisions
- an audit log

No PC action is executed by the current brain. Tool execution begins only after the complete AI/model architecture has been developed, tested and evaluated.

## Run locally

From the repository root:

```bash
python -m pytest -q
python train_xor.py
python train_tiny_lm.py
python experiments/scaled_lm_diagnostics.py
python experiments/train_scaled_lm.py
python experiments/heldout_generalization.py
python experiments/scientific_scaling.py
```

## Roadmap

The current repository has both the detailed AI/model foundation and a complete basic cognitive architecture. The next development phase is **depth, not more disconnected skeleton layers**: upgrade each basic layer into a capable implementation while preserving the verified interfaces.

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

48. ✅ Train and benchmark the PC-oriented scaled model
49. ✅ Compare capacity against the tiny baseline
50. ✅ Evaluate data/capacity/compute trade-offs
51. ✅ Improve training data and language evaluation
52. ✅ Compare training optimizers under a matched experimental budget
53. ✅ Separate held-out evaluation from training/validation measurements
54. ✅ Record a reproducible scaling protocol and compute-budget proxy

### Phase 11 — Full Brain Integration

55. ✅ Connect language generation to the cognitive state boundary
56. ✅ Connect observation and long-term retrieval to response generation
57. ✅ Connect goals, planning, verification and recovery to one brain interface
58. ✅ Keep inference non-autograd and CPU-friendly
59. ✅ Add integration regression tests
60. ✅ Freeze the pre-tool AI/model architecture for final validation

### Phase 12 — Basic Complete Architecture

61. ✅ Unified basic perception/memory/reasoning/planning/action loop
62. ✅ Unified verification/reflection/learning loop
63. ✅ Basic hypothesis and experiment interfaces
64. ✅ Basic knowledge graph
65. ✅ Research and multimodal boundaries
66. ✅ Basic autonomous orchestration boundary

### Phase 13 — PC Automation / Tools — intentionally deferred

Only after the AI/model architecture is complete and final validation is run:

67. ⬜ File tools
68. ⬜ Terminal tools
69. ⬜ Application/browser tools
70. ⬜ Controlled keyboard/mouse interaction
71. ⬜ Tool selection and execution verification
72. ⬜ Emergency stop / rollback where possible

TARA stays intentionally small and inspectable. The objective is to understand and implement the core mechanisms ourselves, validate them mathematically and empirically, then connect them into a useful reasoning-and-tool system rather than imitate frontier-model scale.
