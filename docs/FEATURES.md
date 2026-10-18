# TARA Feature Catalog

**TARA — Tiny Artificial Reasoning Architecture**

This catalog separates the normal user workflow from source-level components and historical experiments. A module or test existing in the repository does **not** prove that the capability is active in every runtime configuration. The default chat workflow uses a configured language-model provider (Ollama for the documented local setup); TARA supplies the surrounding application and orchestration.

## Status key

- **Current documented workflow** — described as the normal way to use TARA on `main`.
- **Source-level capability** — corresponding code and/or tests exist in the repository; runtime integration and configuration may vary.
- **Optional integration** — requires setup, credentials, or a compatible configuration.
- **Historical / experimental** — engineering code and tests exist, but this is not required for normal chat and should not be treated as a production capability without fresh validation.
- **Proposed / branch-dependent** — do not assume it is available on `main` until the relevant changes are merged and tested.

## 1. Chat and model providers

| Feature | Status | Notes |
|---|---|---|
| Interactive chat runtime | Current documented workflow | Start with `python -m src`; use `/help` and `/exit`. |
| Local model inference through Ollama | Current documented workflow | Example setup uses `ollama pull llama3.2:3b`. TARA does not need to train its own model for this workflow. |
| Configurable inference backend | Source-level capability | Provider configuration is documented via `TARA_LLM_BACKEND` and `.env.example`; check the active checkout for exact supported values. |
| Groq provider integration | Source-level / configuration-dependent | Provider code and tests exist; actual use depends on configuration and credentials. |
| Optional ElevenLabs text-to-speech | Optional integration | Requires provider configuration and credentials; keep secrets in environment variables. |

## 2. Reasoning, planning, and orchestration

| Feature | Status | Notes |
|---|---|---|
| Structured reasoning | Source-level capability | Dedicated reasoning modules and tests exist. |
| Planning and advanced planning | Source-level capability | Planning components are present; behavior depends on runtime wiring. |
| Agent loop / unified cognitive loop | Source-level capability | Coordinates cognitive stages in code; not every component is necessarily enabled in every setup. |
| Autonomous task orchestration | Source-level capability / experimental | Orchestration, scheduling, resource, and goal-progress modules exist. Validate boundedness and recovery before relying on unattended execution. |
| Reflection and experience tracking | Source-level capability | Reflection and experience-learning modules and tests exist. |
| Outcome/goal progress tracking | Source-level capability | Dedicated goal-progress and reflection components are present. |

## 3. Memory, perception, and world context

| Feature | Status | Notes |
|---|---|---|
| Memory primitives and retrieval | Source-level capability | Memory modules and retrieval tests exist. Persistence behavior should be checked in the chosen runtime. |
| Cognitive memory and consolidation | Historical / experimental | Memory consolidation and continual-learning components exist; not required for standard Ollama chat. |
| Perception primitives | Source-level capability | Perception modules and tests exist. |
| Temporal perception | Source-level capability | Temporal context module and tests exist. |
| Multimodal perception | Source-level capability / integration-dependent | Code and tests exist; supported inputs and live integration should be verified for the active runtime. |
| World state and event routing | Source-level capability | World-state and event-router modules exist; runtime use is configuration-dependent. |
| Embeddings | Source-level capability | Embedding module and tests exist; verify the configured backend and intended use. |

## 4. Tools and safety boundaries

| Feature | Status | Notes |
|---|---|---|
| File and input tools | Source-level capability | Tool modules and tests exist. |
| Terminal/sandbox execution | Source-level capability; use cautiously | Terminal tools and sandbox runner are present. Execution should remain explicitly authorized and bounded. |
| Capability-based tool selection | Source-level capability | Tool-intelligence module and tests exist. |
| Safety checks and permission boundaries | Source-level capability | Safety module and tests exist. Model-generated text must not itself authorize an external action. |
| Public read-only API commands (GitHub, Wikipedia, weather, Hacker News, Stack Exchange) | Proposed / branch-dependent | The README describes these as part of open changes. Do not advertise them as available on `main` until merged and validated. |

## 5. Learning, neural-network, and experiment tooling

These are engineering/experimental areas, not requirements for normal chat with an existing model.

| Feature | Status | Notes |
|---|---|---|
| Custom autograd and neural-network primitives | Historical / experimental | Autograd, layers, activations, attention, optimizers, losses, and gradient checks have source modules/tests. |
| Transformer and language-model experiments | Historical / experimental | Tokenizer, generation, model, and training modules exist. |
| Dataset acquisition, preparation, and audit | Historical / experimental | Dataset registry/pipeline/audit and related tests exist; review data licensing and provenance before use. |
| Training, validation, checkpoints, and resume | Historical / experimental | Code and tests exist; not required to run the default chat workflow. Validate current branch behavior before using training commands. |
| Capability benchmarks and evaluation | Historical / experimental | Benchmark and evaluation modules exist. Scores only support the specific tested tasks and setup. |
| Algorithm discovery / Polyglot experiments | Historical / experimental | Algorithm lab/discovery and Polyglot submodules/tests exist; treat claims as experimental until independently benchmarked. |
| Scientific experiments and architecture optimization | Historical / experimental | Experiment and architecture-optimization code exists; changes should be reproducible and regression-gated. |
| Self-improvement proposals | Historical / experimental | Improvement/verification modules exist; proposed changes should never auto-promote without tests, safety checks, and explicit gates. |

## 6. Developer quality and operations

| Feature | Status | Notes |
|---|---|---|
| Automated tests | Current engineering workflow | The repository contains a broad pytest suite; run `pytest` and inspect CI on the branch you intend to use. |
| CI workflows | Current engineering workflow | GitHub Actions workflows exist; passing CI is required before describing a change as validated. |
| Repository audit / self-test utilities | Source-level capability | Audit and self-test modules exist; use their output as diagnostics, not proof of general intelligence. |
| Runtime lifecycle and health diagnostics | Source-level capability | Core runtime modules cover lifecycle/health concepts; confirm actual configuration and observable behavior. |
| Exception taxonomy and reporting | Proposed / branch-dependent | Exception-handling work is under review in a separate PR; do not assume all modules are wired to it yet. |

## What TARA is — and is not

- TARA is a modular reasoning/agent runtime around a configured language model.
- Running chat with Ollama is **inference**, not training.
- Repository modules and historical phase notes are not proof that every feature is enabled, robust, or production-ready.
- Autonomous execution, self-improvement, training, and external tools need bounded permissions, clear logs, reproducible tests, and human approval at consequential boundaries.

## Related documentation

- [README / quick start](../README.md)
- [Feature suggestions and implementation roadmap](FEATURE_SUGGESTIONS.md)
