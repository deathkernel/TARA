# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a research-first neural-network project built from mathematical and implementation fundamentals. The goal is a highly capable, inspectable artificial reasoning system that connects a neural language core with perception, memory, structured reasoning, advanced planning, intelligent tools, algorithm discovery, self-improvement, reflection, autonomous tasks, scheduling, temporal context and an explicit world model.

## Phase 26 — Advanced Self-Improvement Complete

Phase 26 implements steps **157–164**:

- **157 Failure analysis** — benchmark failures become structured categories, severity and targeted optimization signals.
- **158 Targeted mutation** — injectable mutation policies can be backed by an LM, evolutionary operator or program-synthesis model.
- **159 Experiment generation** — bounded candidate variants are deduplicated before execution.
- **160 Regression benchmarking** — every experiment is re-run through the deterministic benchmark boundary.
- **161 Multi-objective improvement scoring** — correctness, speed, compactness and regression safety are combined into measurable utility.
- **162 Promotion gate** — correctness regressions are rejected before a candidate can replace the baseline.
- **163 Iterative improvement** — successful candidates become the next baseline for another bounded cycle.
- **164 Brain integration** — `TARABrain` exposes improvement-cycle and iterative-improvement APIs.

`src/self_improvement_lab.py` makes the loop explicit: **failure → analysis → mutation → experiment → benchmark → regression gate → measured promotion**. Execution remains behind the existing polyglot sandbox/benchmark boundary. The architecture is ready for learned mutation and ranking policies, but no model is claimed to have trained itself merely by enabling this code.

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
  ├─ Failure Analysis
  ├─ Targeted Mutation
  ├─ Experiment Generation
  ├─ Regression Benchmarking
  ├─ Multi-objective Scoring
  ├─ Promotion Gate
  └─ Iterative Improvement
        ↓
Reflection → Goal Progress → Learning / Continual Learning
        ↓
Autonomous Tasks → Scheduler → Resource Budget
        ↓
Neural Language Core
```

The improvement stack separates **proposal** from **execution** and **promotion**. A generated mutation is never treated as correct because it looks plausible: deterministic tests and a regression gate must approve it first.

## Roadmap

- Phase 21 — Perception & Temporal Awareness: **122–127 complete**
- Phase 22 — Structured Reasoning: **128–134 complete**
- Phase 23 — Advanced Planning: **135–141 complete**
- Phase 24 — Tool Intelligence: **142–148 complete**
- Phase 25 — Algorithm Discovery: **149–156 complete**
- Phase 26 — Self-Improvement: **157–164 complete**
- Phase 27 — Continual Learning: **165–172**
- Phase 28 — Persistent Cognitive Memory: **173–180**
- Phase 29 — Learning From Experience: **181–188**
- Phase 30 — Multimodal Perception: **189–197**
- Phase 31 — Autonomous Research: **198–205**
- Phase 32 — Scientific Experiment Engine: **206–213**
- Phase 33 — Architecture Self-Optimization: **214–221**
- Phase 34 — Unified Cognitive Loop: **222–230**
- Phase 35 — TARA Core Integration: **231–240**
