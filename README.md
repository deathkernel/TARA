# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a research-first neural-network project built from mathematical and implementation fundamentals. The goal is a small, inspectable artificial reasoning system that can connect a neural language core with perception, memory, reasoning, planning, reflection and controlled PC tools.

## Phase 19 — Basic Persistent Scheduling and Resource Awareness Complete

Phase 19 contains a **basic implementation of all planned scheduling steps (110–115)**. TARA can persist scheduled task metadata, identify due work, apply deterministic priority ordering, enforce simple resource budgets, restore pending tasks after restart, and feed due work into the bounded Phase 18 orchestrator.

- Persistent task store: JSONL-backed scheduled task metadata.
- Due-task detection: tasks can be scheduled for a future ISO timestamp.
- Resource awareness: explicit maximum-task and retry budgets bound execution.
- Deterministic scheduling: due tasks are ordered by priority and task ID.
- Restart continuity: persisted tasks can be loaded by a new scheduler instance.
- Orchestrator integration: due tasks execute through the existing bounded host executor boundary.

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
Persistent Task Store → Scheduler
       ↓
Resource Budget → Task Queue → Autonomous Orchestrator
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

Scheduling and resource budgeting do not create permissions. A scheduled task can execute only through the callable supplied by the host application. The existing Tool Controller remains the permission boundary for PC actions. Execution remains bounded by explicit task and resource limits.

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

### Phase 18 — Autonomous Task Orchestration

104. ✅ Explicit task queue
105. ✅ Dependency-aware scheduling
106. ✅ Priority-based task selection
107. ✅ Bounded execution loop
108. ✅ Retry / failure recovery
109. ✅ Brain integration + stop/resume control

### Phase 19 — Persistent Scheduling / Resource Awareness

110. ✅ Persistent task store
111. ✅ Due-task scheduling
112. ✅ Resource budgets
113. ✅ Deterministic scheduler ordering
114. ✅ Restart/recovery continuity
115. ✅ Orchestrator integration

Phase 19 is complete at the **basic architecture level**. Future phases can deepen resource accounting, calendar/time semantics, distributed scheduling, richer task persistence, and learned scheduling policies without removing the explicit control boundaries.
