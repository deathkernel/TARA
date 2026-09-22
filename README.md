# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a research-first neural-network project built from mathematical and implementation fundamentals. The goal is a highly capable, inspectable artificial reasoning system that connects a neural language core with perception, memory, structured reasoning, advanced planning, intelligent tools, reflection, autonomous tasks, scheduling, temporal context and an explicit world model.

## Phase 24 — AI Tool Intelligence Complete

Phase 24 implements steps **142–148**:

- **142 Tool capability registry** — explicit capability representations and fallback metadata.
- **143 Tool precondition checking** — fact-backed and injectable predicate checks before selection.
- **144 Tool selection reasoning** — contextual utility scoring using capability coverage, reliability, risk, latency and cost.
- **145 Tool-result interpretation** — structured attempt/result representation for downstream diagnosis.
- **146 Tool failure diagnosis** — transient, authorization, precondition and execution classification with confidence.
- **147 Safer tool fallback** — declared fallback validation plus contextual alternative selection.
- **148 Brain integration** — `TARABrain` can register capabilities, choose tools and recover from failed tool attempts.

The new `src/tool_intelligence.py` layer is model-agnostic: its utility scorer can later be replaced or augmented by a learned policy trained from verified tool outcomes. It does not execute tools itself; actual execution remains behind the existing `ToolController` safety boundary.

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
  ├─ Goal Decomposition
  ├─ Dependency Graph
  ├─ Validation
  ├─ Alternative Plans
  ├─ Cost / Risk Evaluation
  └─ Failure-driven Revision
        ↓
AI Tool Intelligence
  ├─ Capability Registry
  ├─ Preconditions
  ├─ Contextual Utility Scoring
  ├─ Tool Selection
  ├─ Failure Diagnosis
  └─ Safe Fallback
        ↓
Tool Controller
        ↓
File / Terminal / App / Browser / Input Tools
        ↓
Observation → Verification → Experience
        ↓
Reflection → Goal Progress → Learning / Continual Learning
        ↓
Autonomous Tasks → Scheduler → Resource Budget
        ↓
Neural Language Core
```

The project intentionally separates **decision intelligence** from **action execution**. AI/ML methods can improve selection and planning, while explicit verification, permissions, bounded execution and emergency-stop mechanisms remain deterministic safety boundaries.

## Roadmap

- Phase 21 — Perception & Temporal Awareness: **122–127 complete**
- Phase 22 — Structured Reasoning: **128–134 complete**
- Phase 23 — Advanced Planning: **135–141 complete**
- Phase 24 — Tool Intelligence: **142–148 complete**
- Phase 25 — Algorithm Discovery: **149–156**
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
