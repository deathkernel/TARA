# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a research-first neural-network project built from mathematical and implementation fundamentals. The goal is a highly capable, inspectable artificial reasoning system that connects a neural language core with perception, memory, structured reasoning, advanced planning, intelligent tools, algorithm discovery, self-improvement, continual learning, reflection, autonomous tasks, scheduling, temporal context and an explicit world model.

## Phase 27 — Advanced Continual Learning Complete

Phase 27 implements steps **165–172**:

- **165 Verified replay orchestration** — learning examples are admitted only when verification is true.
- **166 Deterministic normalization/deduplication** — examples receive stable fingerprints and provenance.
- **167 Balanced replay sampling** — bounded importance-weighted sampling maintains domain coverage.
- **168 Stability/plasticity measurement** — improvement and forgetting are measured separately.
- **169 Catastrophic-forgetting diagnostics** — retained-skill degradation is quantified against a configurable budget.
- **170 Evidence-gated promotion** — learning is rejected when forgetting exceeds the safety threshold or measurable progress is absent.
- **171 Replay manifest generation** — reproducible replay batches can be fingerprinted and persisted.
- **172 Brain integration** — `TARABrain` can build verified replay batches and evaluate learning promotion.

`src/continual_learning_engine.py` adds an ML-oriented stability/plasticity layer without pretending that dataset construction equals model training. Actual parameter updates remain an explicit training operation. This keeps learning auditable: **verified data → replay sampling → train externally → evaluate retained skills → promotion gate**.

## Architecture

```text
Input / Perception
        ↓
Working Memory → Long-Term Memory
        ↓
World Model → Temporal Context
        ↓
Structured Reasoning
        ↓
Advanced Planning
        ↓
AI Tool Intelligence
        ↓
Algorithm Discovery
        ↓
Self-Improvement Lab
        ↓
Continual Learning
  ├─ Verified Replay
  ├─ Deduplication / Provenance
  ├─ Importance-weighted Sampling
  ├─ Domain Balancing
  ├─ Stability / Plasticity
  ├─ Forgetting Diagnostics
  └─ Evidence-gated Promotion
        ↓
Reflection → Goal Progress
        ↓
Autonomous Tasks → Scheduler → Resource Budget
        ↓
Neural Language Core
```

The continual-learning layer deliberately separates **learning data selection** from **parameter updates**. A future learned replay policy can optimize sampling, but verification and forgetting gates remain explicit.

## Roadmap

- Phase 21 — Perception & Temporal Awareness: **122–127 complete**
- Phase 22 — Structured Reasoning: **128–134 complete**
- Phase 23 — Advanced Planning: **135–141 complete**
- Phase 24 — Tool Intelligence: **142–148 complete**
- Phase 25 — Algorithm Discovery: **149–156 complete**
- Phase 26 — Self-Improvement: **157–164 complete**
- Phase 27 — Continual Learning: **165–172 complete**
- Phase 28 — Persistent Cognitive Memory: **173–180**
- Phase 29 — Learning From Experience: **181–188**
- Phase 30 — Multimodal Perception: **189–197**
- Phase 31 — Autonomous Research: **198–205**
- Phase 32 — Scientific Experiment Engine: **206–213**
- Phase 33 — Architecture Self-Optimization: **214–221**
- Phase 34 — Unified Cognitive Loop: **222–230**
- Phase 35 — TARA Core Integration: **231–240**
