# TARA upstream integration architecture

TARA does not vendor TensorFlow, JAX, Bazel, Error Prone, OSS-Fuzz, Gemma,
Microsoft Phi, ONNX Runtime, Semantic Kernel, or the OpenAI repositories.

Instead, TARA treats them as an **upstream capability mesh**:

1. **Manifest** — config/upstream_integrations.json defines every upstream,
   its Git ref, role, capability set and optional package.
2. **Lock** — config/upstream_lock.json records the exact upstream commit that
   TARA last validated.
3. **Adapters** — src/upstream/adapters.py lazily imports optional runtimes.
4. **Model layer** — Gemma, Phi and gpt-oss can be consumed through standard
   inference interfaces instead of copying model source trees.
5. **Evaluation layer** — OpenAI Evals and gpt-oss evals can feed evidence into
   TARA's existing benchmark and regression gates.
6. **Build/security layer** — Bazel, Error Prone and OSS-Fuzz belong in CI and
   security workflows rather than the neural runtime.
7. **Automatic update loop** — GitHub Actions resolves every tracked upstream
   every six hours. If a source moves, it opens a PR containing the new lock.
8. **Dependency isolation** — optional Python packages live in
   requirements-integrations.txt rather than the core requirements file.

## Coordination model

    TARA Cognitive Core
           |
    Capability Router
       /    |     \
    runtime models tools
      |       |      |
    TF/JAX  Gemma/Phi  OpenAI SDK
    ONNX    gpt-oss    Whisper
           |
       Evaluation Gate
       /            \
    OpenAI Evals    gpt-oss evals
           |
    Build / security: Bazel + Error Prone + OSS-Fuzz

All external projects remain independently versioned.

## Auto-update semantics

upstream change
  -> detect
  -> resolve commit SHA
  -> update TARA lock
  -> run compatibility tests
  -> open PR
  -> merge
  -> active TARA environment can be upgraded

The updater never blindly executes external repository code. This is important
because a moving upstream ref can contain breaking API changes or generated
code that should not become part of TARA without validation.

## Current mapping

- TensorFlow/JAX: numerical and training backends.
- Bazel: reproducible builds and tests.
- Error Prone: Java static analysis.
- OSS-Fuzz: continuous fuzzing and security feedback.
- Gemma: Google DeepMind model family.
- Phi: Microsoft model family; the tracked GitHub source is PhiCookBook while
  actual weights are obtained from Microsoft's model distribution.
- ONNX Runtime: cross-platform inference/training runtime.
- Semantic Kernel: optional orchestration bridge. Microsoft now describes
  Agent Framework as its successor, so this remains compatibility-oriented.
- gpt-oss: OpenAI open-weight reasoning model family.
- Whisper: speech recognition and translation input.
- OpenAI Evals: evaluation framework.
- openai-python/openai-node: API SDK bridges.
- tiktoken: optional BPE tokenizer.
- OpenAI Cookbook: integration recipes and documentation.

TARA remains the owner of cognition, memory, planning, world state, safety and
learning decisions. Upstream projects provide specialized engines.
