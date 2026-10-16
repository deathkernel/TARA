# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a modular Python reasoning project. Its architecture separates language-model inference from perception, planning, memory, tool selection, reflection, world-state context, verification, and task orchestration. This README serves two purposes: a quick-start guide for the current runtime and a progress ledger for the project's engineering phases.

> **Runtime rule:** the normal chat workflow uses Ollama (or another configured inference provider). It does not require training TARA's own language model. Dataset, training, and checkpoint experiments listed in the history below are historical engineering work, not startup steps.

## Current status at a glance

| Area | Tracking status | Notes |
|---|---|---|
| Local chat | Current workflow | Ollama supplies language-model inference; TARA provides the application/runtime around it. |
| Cognitive architecture | Architecture area | Perception, reasoning, planning, tools, reflection, world context, and orchestration are separate concerns. |
| Public read-only API tools | In progress on an isolated branch | A bounded library client covers GitHub, Wikipedia, Open-Meteo, Hacker News, and Stack Exchange; chat command routing remains separate and is not shipped on `main`. |
| Voice output | Optional integration | Requires local provider configuration and credentials. |
| Training/data/checkpoints | Legacy/historical | Not required to run chat with Ollama. Do not use old training commands as current setup instructions. |
| Tests/CI | Required validation | Run the test suite and check CI before considering code changes ready. |

## Run locally

Install dependencies, make sure Ollama is running, and install the model you want to use:

```bash
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
python -m src
```

Set `TARA_LLM_BACKEND` as needed; see [`.env.example`](.env.example) for the supported settings in your checkout. Use `/help` in chat to see the commands available in the version you are running.

## Runtime and architecture map

```text
User input
   |
   v
TARA chat / core runtime
   |-- context and perception
   |-- reasoning and planning
   |-- memory / world-state components (where enabled)
   |-- explicit, authorized tools
   |   `-- proposed read-only public API client (allowlist, bounded responses, untrusted results)
   |-- verification and response handling
   |
   v
Configured language-model provider (Ollama for local setup)
   |
   v
Response to user
```

The provider generates language; it does not replace TARA's orchestration or automatically authorize tool actions. Availability of a module in the repository does not prove that it is enabled in every runtime configuration.

## Current and proposed chat commands

The exact commands depend on the branch you run. The baseline chat interface supports `/help` and `/exit`; other commands are branch-dependent.

The open public-API chat work proposes the following explicit, read-only commands:

```text
/api github <query>
/api wiki <query>
/api weather <city>
/api hn <query>
/api stackoverflow <query>
```

The library-level API client is separate from command routing. These commands remain unavailable on `main` until the command-routing change is reviewed and merged; the client rejects redirects and marks remote results as untrusted.

## Engineering principles

1. **Inference is not training.** Running a model with Ollama is a normal inference workflow; it does not train the model.
2. **Measure before making capability claims.** Implemented code, executed behavior, and measured performance are different evidence levels.
3. **Explicit permissions.** Model-generated text does not grant permission to run tools or perform external actions.
4. **Bounded integrations.** External requests should have allowlists, timeouts, response limits, validation, and clear error handling.
5. **Verification and provenance.** Where supported, preserve source URLs, test evidence, reproducible inputs, and clear status labels.
6. **No false completion claims.** A phase means implementation/documentation work was recorded; it is not proof of general intelligence or production readiness.

## Project progress ledger

This index preserves the project's phase history so progress remains trackable. A phase marked "implemented/documented" reflects the recorded engineering history; it does not guarantee that every subsystem is active in the current Ollama chat path. Where no dedicated historical specification was preserved, that limitation is stated rather than filled with guesses.

| Phase | Focus | Recorded status |
|---|---|---|
| 1 | Project/neural architecture foundation | Historical details not preserved as a standalone phase document |
| 2 | Early model/data foundation | Historical details not preserved as a standalone phase document |
| 3 | Memory primitives | Documented |
| 4 | Early architecture/data transition | Standalone specification not preserved |
| 5 | Perception primitives | Implemented + tested in project history |
| 6 | Cognitive integration layer | Implemented + tested in project history |
| 7 | Agent loop | Implemented + tested in project history |
| 8 | Safety foundation | Implemented + tested in project history |
| 9 | Architecture consolidation | Standalone specification not preserved |
| 10 | Architecture consolidation | Standalone specification not preserved |
| 11 | Architecture consolidation | Standalone specification not preserved |
| 12 | Architecture consolidation | Standalone specification not preserved |
| 13 | Training/data transition | Standalone specification not preserved |
| 14 | Training ↔ Brain integration | Historical implementation documented |
| 15 | Continual learning + memory consolidation | Historical implementation documented |
| 16 | Controlled file/tool stack | Implemented + tested in project history |
| 17 | Reflection + experience | Implemented + tested in project history |
| 18 | Autonomous task orchestration | Implemented + tested in project history |
| 19 | Resources + persistent scheduling | Implemented + tested in project history |
| 20 | World model + event context | Implemented + tested in project history |
| 21 | Confidence-aware perception + temporal awareness | Implemented + documented in project history |
| 22 | Structured reasoning | Implemented + tested in project history |
| 23 | Reasoning-system transition | Standalone specification not preserved |
| 24 | Tool intelligence | Implemented + tested in project history |
| 25 | Integration transition | Standalone specification not preserved |
| 26 | Verified self-improvement | Implemented + tested in project history |
| 27 | Continual-learning engine | Historical implementation documented; not needed for normal Ollama chat |
| 28 | Persistent cognitive memory | Documented historically; verify current branch behavior before relying on persistence |
| 29 | Cognitive-system transition | Standalone specification not preserved |
| 30 | Multimodal perception | Implemented + integrated in project history |
| 31 | Autonomous research | Implemented + tested in project history |
| 32 | Scientific experiment engine | Implemented + tested in project history |
| 33 | Architecture self-optimization | Implemented + tested in project history |
| 34 | Unified cognitive loop | Implemented + tested in project history |
| 35 | TARA Core runtime | Implemented + tested in project history |
| 36 | Repository audit + canonical CI | Implemented + tested in project history |
| 37 | Intelligence measurement and evidence-gated learning experiments | Historical sub-phases 37.1–37.12 recorded below; not part of normal Ollama chat setup |

### Phase notes

- **Phases 1–4 — Foundations:** early neural/model/data and memory foundations. The repository does not preserve standalone specifications for every early phase.
- **Phases 5–8 — Cognitive foundation:** perception primitives, cognitive integration, agent loop, and safety boundaries.
- **Phases 9–13 — Consolidation:** transition work toward the explicit training/data path; detailed standalone phase briefs are not preserved.
- **Phase 14 — Training/Brain integration (historical):** reproducible language-model training, deterministic splits, sequence preparation, checkpoint save/load/resume, validation-loss measurement, and runtime integration were recorded.
- **Phase 15 — Continual learning (historical):** controlled replay, memory consolidation, evidence-gated learning, and separation of knowledge preparation from training.
- **Phase 16 — Controlled tools:** allow-listed file/tool interfaces and explicit execution boundaries.
- **Phase 17 — Reflection:** experience/outcome records, success/failure signals, reflection notes, and goal-progress tracking.
- **Phase 18 — Task orchestration:** dependencies, priorities, bounded retries, verification feedback, and stop/resume controls.
- **Phase 19 — Resources/scheduling:** resource-aware scheduling and persistence primitives.
- **Phase 20 — World model:** world-state, event context, and event routing.
- **Phase 21 — Temporal perception:** confidence-aware observations and temporal context.
- **Phase 22 — Structured reasoning:** explicit reasoning state and verification.
- **Phase 24 — Tool intelligence:** capability-based tool selection and recovery among already-authorized tools.
- **Phase 26 — Self-improvement:** benchmark-driven proposals, experiments, evidence checks, and promotion gates.
- **Phases 27–28 — Learning/memory history:** continual-learning and persistent cognitive-memory work was documented; current runtime behavior must be checked against the active branch.
- **Phase 30 — Multimodal perception:** multimodal perception core and brain integration.
- **Phase 31 — Research:** bounded autonomous research workflow.
- **Phase 32 — Scientific experiments:** explicit experiment definitions, reproducibility controls, tests, and integration.
- **Phase 33 — Architecture optimization:** candidate architecture changes as explicit experiments.
- **Phase 34 — Unified cognitive loop:** orchestration across perception/context, memory, reasoning, planning, tools, verification, and reflection.
- **Phase 35 — Core runtime:** lifecycle, component discovery, health diagnostics, event tracing, safe shutdown, and runtime APIs.
- **Phase 36 — Repository validation:** static repository auditing and canonical CI.
- **Phase 37 — Measurement/learning experiments:** a historical set of benchmarks, training hardening, checkpoint evaluation, and evidence-gated experiment workflows. These are not required to run an existing model through Ollama.

### Phase 37 sub-phase ledger (historical)

| Sub-phase | Recorded work |
|---|---|
| 37.1 | Deterministic capability benchmark cases, category scoring, failure capture, fingerprints, and regression gates |
| 37.2 | Dataset quality, duplicate, split-leakage, and contamination-marker audits |
| 37.3 | Historical training controls: gradient accumulation, learning-rate schedule, early stopping, metrics, and resume compatibility |
| 37.4 | Historical Transformer options: depth, RMSNorm, dropout, weight tying, and architecture metadata |
| 37.5 | Reproducible training experiment identity, manifests, metrics, and provenance |
| 37.6 | Capability benchmark categories: coding, reasoning, memory, planning, tools, algorithms, and learning |
| 37.7 | Failure analysis, targeted proposals, regression protection, and evidence-gated promotion |
| 37.8 | Checkpoint-based capability evaluation, deterministic inference, and evaluation fingerprints |
| 37.9 | Evaluation/training continuity and fail-closed handling for generated-code execution |
| 37.10 | Baseline/candidate comparison, capability deltas, regression protection, and experiment fingerprints |
| 37.11 | Historical end-to-end learning experiment workflow with explicit evaluation and promotion gates |
| 37.12 | Historical post-training evidence consolidation |

**Important:** the phase ledger is project history, not a set of instructions to run. No dataset, preflight, training, checkpoint, or learning-experiment commands are listed as current setup steps. For normal local use, start Ollama and run TARA as described above.

## Active work / review queue

### Bulk CI regression batch — 2026-10-16

- [x] Fix unified cognitive-loop tool selection to treat `ToolSelection.selected` as the selected tool-name string.
- [x] Convert lightweight candidate objects into explicit JSON-safe fields before storing learning records.
- [x] Budget scheduler execution steps for each selected task's bounded retries.
- [x] Separate Stack Exchange provider-error and missing-items regression fixtures.
- [x] Correct the PNG test fixture to use a valid PNG signature.
- [x] Validate Open-Meteo city input, geocoding result shapes, finite coordinate bounds, and forecast `current` payloads.
- [x] Add regression tests for malformed Open-Meteo responses and document the fail-closed validation boundary.
- [ ] Verify focused regressions and the full Python/Rust CI on the exact batch commit before treating the batch as validated.

Keep this section updated as PRs change; do not mark proposed features as shipped until they are merged and validated.

Keep this section updated as PRs change; do not mark proposed features as shipped until they are merged and validated.

- [ ] Review the proposed dataset/training cleanup only after verifying it preserves `checkpoints/`, `outputs/`, `data/conversation_v1.jsonl`, `data/conversation_v1.jsonl.manifest.json`, and `test_voice.py`; do not retire any integration without explicit approval.
- [ ] Review the integration-preserving read-only public API toolkit branch and its security tests; redirect protection is covered by regression tests, but the work remains unshipped until exact-commit CI passes.
- [ ] Review the explicit chat commands for GitHub, Wikipedia, weather, Hacker News, and Stack Exchange.
- [ ] Confirm the main-branch README and command list match the actual runtime after the PR stack is resolved.
- [ ] Run the canonical CI/test suite and record its result before calling the work complete.

### Public API input-validation follow-up — 2026-10-16

- [x] Reject non-finite/boolean request timeouts and pacing intervals, and non-integer response byte limits.
- [x] Apply hard upper bounds to request timeout, response bytes, and per-host pacing to prevent pathological configuration values.
- [x] Normalize malformed URL parsing and request-construction failures into the public API's fail-closed error type.
- [x] Validate public search query types and require exact integer result limits before making a request.
- [x] Add deterministic regression coverage for invalid configuration, malformed URLs, query types, and limit types.
- [ ] Verify the focused API tests and full Python/Rust CI on the exact batch commit.

### Full-suite regression follow-up — 2026-10-16

- [x] Preserve ready planner steps across cognitive-loop iterations and keep tool selection consistent with the string-based `ToolSelection.selected` contract.
- [x] Correct the clean-baseline/category-regression test fixture, the dataset-viewer minimum-length fixture, evaluation evaluator setup, and embedding parameter-count expectation.
- [x] Make checkpoint adapter entry points patchable while retaining lazy imports and safe tensor-only checkpoint loading.
- [x] Let the model capability runner use TARA's native `forward_numeric` interface without requiring PyTorch-only methods on test doubles.
- [x] Replace host-process execution in the self-improvement test with an injected deterministic fake executor.
- [x] Use combined absolute/relative gradient-check tolerances to avoid false failures on near-zero gradients.
- [ ] Re-run the 431-test suite, static repository audit, and Rust tests on the exact batch commit.

## Optional voice output

TARA may be configured for ElevenLabs voice output where the corresponding integration is present. Keep credentials in local environment variables and never commit API keys or secrets. Check the active branch's implementation and current SDK documentation before relying on specific parameters.

## Development checks

Run tests with:

```bash
pytest
```

Prefer mocked responses for deterministic unit tests. Live integrations may require credentials and network access. CI must pass before a change is considered ready to merge.

## Documentation policy

- Keep current run instructions short and executable.
- Keep historical phases, architecture decisions, and completed/in-progress work in the progress ledger.
- Clearly label retired pipelines as historical instead of presenting them as current commands.
- Track proposals separately from merged/current functionality.
- Update the ledger when a phase or PR changes status.
