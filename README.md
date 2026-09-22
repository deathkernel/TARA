# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a research-first neural-network project built from mathematical and implementation fundamentals. The goal is a highly capable, inspectable artificial reasoning system that connects a neural language core with perception, memory, structured reasoning, advanced planning, intelligent tools, algorithm discovery, self-improvement, continual learning, reflection, autonomous tasks, scheduling, temporal context and an explicit world model.

## Phase 37.3 — Training Hardening Complete

Phase 37.3 connects the training-control layer to the real PyTorch pipeline:

- **Gradient accumulation** — optimizer updates can aggregate multiple micro-batches while keeping the configured batch size bounded.
- **Warmup + cosine decay** — learning rate is scheduled per optimizer update with a configurable minimum ratio.
- **Validation early stopping** — validation loss drives explicit patience/min-delta stopping decisions.
- **Append-only experiment tracking** — training metrics are persisted as JSONL with a stable SHA-256 fingerprint.
- **Checkpoint continuity** — checkpoint format 2 stores hardening configuration, scheduler progress, early-stopping state and metric fingerprint.
- **Resume safety** — incompatible dataset fingerprints, model configuration or training controls are rejected rather than silently mixing experiments.
- **CLI controls** — `train_algorithm_lm.py` exposes accumulation, scheduler, early-stopping and metrics-path options.

Actual model training remains an explicit offline operation. This phase implements the controls; it does not claim that a new model has already been trained.

## Phase 37.2 — Dataset Intelligence Audit Complete

Phase 37.2 adds a deterministic audit boundary before training:

- **Dataset quality** — detects empty and unusually short examples.
- **Duplicate analysis** — fingerprints normalized examples and reports duplicate records.
- **Split leakage detection** — compares train/validation/test fingerprints and raises an error-level issue for overlap.
- **Evaluation contamination signals** — configurable markers such as ground-truth/test-answer language are surfaced for review.
- **Reproducibility** — dataset audit reports receive stable SHA-256 fingerprints.
- **Non-destructive operation** — auditing never modifies the source corpus.
- **JSONL support** — prepared corpora can be audited directly from JSONL files.

The audit is a signal, not a proof of dataset quality: semantic contamination and memorization require deeper analysis than string fingerprints alone.

## Phase 37.1 — Intelligence Baseline Benchmark Started

Phase 37 begins with measurement before model changes. This avoids treating architecture size or training loss as a proxy for intelligence.

- **Benchmark case model** — each task has an ID, capability category, expected result, optional custom scorer and weight.
- **Deterministic scoring** — exact and normalized-text scorers are provided, with strict `[0, 1]` validation for custom metrics.
- **Category aggregation** — weighted overall and per-capability scores are reported.
- **Failure capture** — solver exceptions become measured failed cases rather than crashing the entire benchmark.
- **Stable fingerprints** — benchmark inputs and measured results receive SHA-256 fingerprints for experiment traceability.
- **Regression gate** — candidate reports are compared against a baseline with explicit overall and category-drop tolerances.
- **Smoke capability suite** — coding, reasoning, memory and planning categories provide a small deterministic starting benchmark.

The benchmark executes only an injected solver. It does not grant permissions, run arbitrary generated programs, or silently train the neural model.

## Phase 36 — Deep Audit & Validation Complete

Phase 36 implements the validation and hardening layer that follows the architecture roadmap:

- **Repository static audit** — Python syntax, required Phase 35 integration files, duplicate pytest workflows and non-deterministic `hash()` usage are checked without executing repository code.
- **Scheduler correctness fix** — failed scheduled tasks are retained instead of being silently deleted; persisted retry budgets are honored within the global resource budget.
- **Continual-learning correctness fix** — replay sampling now guarantees domain coverage whenever the replay budget can accommodate all domains, then uses importance-weighted sampling for remaining capacity.
- **Regression tests** — scheduler failure retention, retry budgeting, replay domain coverage and repository-audit rules are covered by tests.
- **CI consolidation** — the duplicate Python full-test workflow was removed; one canonical CI workflow now runs static audit, the Python suite, smoke experiments and Rust tests.
- **Audit boundary** — the phase is deterministic and dependency-light; it does not execute arbitrary generated code during static inspection.

Phase 36 is a hardening milestone, not a claim that every runtime behavior has been exhaustively proven. Full pytest/Rust execution is delegated to the repository's GitHub Actions environment.

## Architecture

```text
Input / Perception
        ↓
Working Memory → Persistent Cognitive Memory
        ↓
World Model → Temporal Context → Structured Reasoning
        ↓
Advanced Planning → AI Tool Intelligence
        ↓
Algorithm Discovery → Self-Improvement → Continual Learning
        ↓
Reflection → Goal Progress → Autonomous Tasks → Scheduler
        ↓
Neural Language Core → Training Controls → Intelligence Benchmarks
```

Memory persistence is deliberately separate from learning, and model training is never silently triggered by runtime orchestration.

## Roadmap

- Phase 21 — Perception & Temporal Awareness: **122–127 complete**
- Phase 22 — Structured Reasoning: **128–134 complete**
- Phase 23 — Advanced Planning: **135–141 complete**
- Phase 24 — Tool Intelligence: **142–148 complete**
- Phase 25 — Algorithm Discovery: **149–156 complete**
- Phase 26 — Self-Improvement: **157–164 complete**
- Phase 27 — Continual Learning: **165–172 complete**
- Phase 28 — Persistent Cognitive Memory: **173–180 complete**
- Phase 29 — Learning From Experience: **181–188 complete**
- Phase 30 — Multimodal Perception: **189–197 complete**
- Phase 31 — Autonomous Research: **198–205 complete**
- Phase 32 — Scientific Experiment Engine: **206–213 complete**
- Phase 33 — Architecture Self-Optimization: **214–221 complete**
- Phase 34 — Unified Cognitive Loop: **222–230 complete**
- Phase 35 — TARA Core Integration: **231–240 complete**
- Phase 36 — Deep Audit & Validation: **complete**
- Phase 37 — Real Intelligence & Training: **37.1 + 37.2 + 37.3 complete**
