# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a research-first neural-network project built from mathematical and implementation fundamentals. The goal is a highly capable, inspectable artificial reasoning system that connects a neural language core with perception, memory, structured reasoning, advanced planning, intelligent tools, algorithm discovery, reflection, autonomous tasks, scheduling, temporal context and an explicit world model.

## Phase 25 — Advanced Algorithm Discovery Complete

Phase 25 implements steps **149–156**:

- **149 Problem understanding** — `ProblemInterpreter` extracts objectives, constraints and optimization targets.
- **150 Algorithm candidate generation** — discovery lab accepts model-backed candidate generators.
- **151 Multi-language candidate generation** — integrates with the existing polyglot candidate abstraction.
- **152 Candidate compilation/execution** — uses the existing bounded polyglot benchmark boundary.
- **153 Correctness verification** — only benchmark-verified candidates enter the learning set.
- **154 Performance benchmarking** — runtime is normalized as an optimization signal.
- **155 Candidate ranking/archive** — multi-objective selection combines runtime, complexity, novelty and diversity, with archive persistence.
- **156 Verified knowledge extraction** — verified benchmark results are converted into reusable knowledge records.

`src/algorithm_lab.py` provides the higher-level research loop while reusing TARA's existing sandbox, polyglot execution, benchmark, novelty, archive and knowledge layers. Generated algorithms are candidates, not assumed truth; novelty is relative to TARA's archive and is not a proof of prior-art novelty.

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
Algorithm Discovery Lab
  ├─ Problem Interpretation
  ├─ Candidate Generation
  ├─ Multi-language Candidates
  ├─ Compile / Execute
  ├─ Correctness Verification
  ├─ Benchmarking
  ├─ Novelty / Diversity Analysis
  ├─ Multi-objective Selection
  └─ Verified Knowledge Extraction
        ↓
Reflection → Goal Progress → Learning / Continual Learning
        ↓
Autonomous Tasks → Scheduler → Resource Budget
        ↓
Neural Language Core
```

The discovery stack deliberately separates **generation** from **verification**. Future phases can replace heuristic candidate scoring with learned ranking policies trained only from verified outcomes.

## Roadmap

- Phase 21 — Perception & Temporal Awareness: **122–127 complete**
- Phase 22 — Structured Reasoning: **128–134 complete**
- Phase 23 — Advanced Planning: **135–141 complete**
- Phase 24 — Tool Intelligence: **142–148 complete**
- Phase 25 — Algorithm Discovery: **149–156 complete**
- Phase 26 — Self-Improvement: **157–164**
- Phase 27 — Continual Learning: **165–172**
- Phase 28 — Persistent Cognitive Memory: **173–180**
- Phase 29 — Learning From Experience: **181–188**
- Phase 30 — Multimodal Perception: **189–197**
- Phase 31 — Autonomous Research: **198–205**
- Phase 32 — Scientific Experiment Engine: **206–213**
- Phase 33 — Architecture Self-Optimization: **214–221**
- Phase 34 — Unified Cognitive Loop: **222–230**
- Phase 35 — TARA Core Integration: **231–240**
