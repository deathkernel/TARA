# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a research-first neural-network project built from mathematical and implementation fundamentals. The goal is a highly capable, inspectable artificial reasoning system that connects a neural language core with perception, memory, structured reasoning, advanced planning, intelligent tools, algorithm discovery, self-improvement, continual learning, reflection, autonomous tasks, scheduling, temporal context and an explicit world model.

## Phase 28 — Persistent Cognitive Memory Complete

Phase 28 implements steps **173–180**:

- **173 Persistent memory records** — episodic and semantic memories have stable IDs, timestamps and provenance.
- **174 Importance and salience** — memory strength combines explicit importance with access history and recency.
- **175 Durable storage** — append-only JSONL persistence supports deterministic save/load and bounded recovery.
- **176 Retrieval indexing** — lexical/token overlap plus salience provides explainable retrieval scoring.
- **177 Memory consolidation** — repeated compatible memories can strengthen a canonical record.
- **178 Conflict resolution** — contradictory memories remain auditable while a deterministic policy selects the active view.
- **179 Forgetting / compaction** — low-value memories can be removed under an explicit retention budget.
- **180 Brain integration** — `TARABrain` exposes persistent memory operations without coupling storage to model training.

## Architecture

```text
Input / Perception
        ↓
Working Memory → Persistent Cognitive Memory
                    ├─ Episodic Memory
                    ├─ Semantic Memory
                    ├─ Salience / Recency
                    ├─ Retrieval Index
                    ├─ Consolidation
                    ├─ Conflict Resolution
                    └─ Bounded Forgetting
        ↓
World Model → Temporal Context → Structured Reasoning
        ↓
Advanced Planning → AI Tool Intelligence
        ↓
Algorithm Discovery → Self-Improvement → Continual Learning
        ↓
Reflection → Goal Progress → Autonomous Tasks → Scheduler
        ↓
Neural Language Core
```

Multimodal Perception feeds text, documents, images, audio and screens into a shared representation before reasoning.

Memory persistence is deliberately separate from learning: storing a memory does not train the model, and model-generated claims are not automatically promoted to trusted semantic knowledge.

## Phase 30 — Multimodal Perception Complete

Phase 30 implements steps **189–197**:

- **189 Text perception** — normalized text becomes a deterministic feature representation.
- **190 Document perception** — document content carries source and size metadata.
- **191 Image perception** — PNG/JPEG metadata and hashed image bytes become a structured image observation.
- **192 Audio perception** — PCM WAV metadata becomes a structured audio observation.
- **193 Screen perception** — existing screen-element summaries become multimodal observations.
- **194 Shared feature space** — modality-specific inputs map into a common bounded vector space.
- **195 Cross-modal attention-style fusion** — confidence-gated softmax weighting combines modalities.
- **196 Unified multimodal context** — fused vector, dominant modality, confidence and fingerprint are exposed downstream.
- **197 Brain integration** — `TARABrain` exposes explicit multimodal perception entry points.

This phase uses deterministic feature hashing and attention-inspired fusion as infrastructure. It does **not** claim that TARA already contains a trained vision, speech-recognition or audio-understanding model.

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
- Phase 31 — Autonomous Research: **198–205**
- Phase 32 — Scientific Experiment Engine: **206–213**
- Phase 33 — Architecture Self-Optimization: **214–221**
- Phase 34 — Unified Cognitive Loop: **222–230**
- Phase 35 — TARA Core Integration: **231–240**
