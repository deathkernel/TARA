# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a research-first neural-network project built from mathematical and implementation fundamentals. The goal is a small, inspectable artificial reasoning system that can connect a neural language core with perception, memory, reasoning, planning, reflection and controlled PC tools.

## Phase 17 — Basic Reflection and Experience Complete

Phase 17 contains a **basic implementation of all planned reflection steps (98–103)**. The layer records outcomes, tracks goal progress, produces deterministic reflections and feeds the result back into the brain without claiming human-like self-awareness.

- Experience memory: bounded JSONL-backed history of actions, outcomes, success and feedback.
- Outcome reflection: computes attempts, successes, failures and success rate.
- Failure analysis: turns recorded failure feedback into a simple lesson.
- Goal progress: tracks bounded verified milestones and completion status.
- Reflection loop: connects experience recording to immediate reflection.
- Brain integration: `TARABrain` can record experiences, reflect and track progress.

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
Tool Controller
       ↓
File / Terminal / App / Browser / Input Tools
       ↓
Observation → Verification
       ↓
Experience Memory → Reflection → Goal Progress
       ↓
Learning / Continual Learning
       ↓
Emergency Stop / Rollback boundary
```

## Safety boundary

Tool access is explicit and host-controlled. File access requires configured roots; terminal execution requires an allow-listed executable; browser/application actions require registered host functions; keyboard/mouse operations require an injected backend. Reflection only records and evaluates supplied outcomes; it does not grant additional permissions.

These controls are defense-in-depth and are not a claim of perfect isolation against hostile code. Truly untrusted code should use a separate OS/container/VM sandbox before production use.

## Roadmap

### Phase 16 — PC Automation / Tools

92. ✅ File tools
93. ✅ Terminal tools
94. ✅ Application/browser tools
95. ✅ Controlled keyboard/mouse interaction
96. ✅ Tool selection and execution verification
97. ✅ Emergency stop / rollback where possible

### Phase 17 — Reflection / Experience

98. ✅ Experience memory
99. ✅ Outcome evaluation
100. ✅ Failure/feedback analysis
101. ✅ Goal progress tracking
102. ✅ Reflection cycle
103. ✅ Brain integration

Phase 17 is complete at the **basic architecture level**. Future phases can deepen learned reflection, richer planning, long-horizon memory, evaluation, and model training without changing the basic boundaries above.
