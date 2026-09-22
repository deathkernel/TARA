# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a research-first neural-network project built from mathematical and implementation fundamentals. The goal is a small, inspectable artificial reasoning system that can connect a neural language core with perception, memory, reasoning, planning, reflection and controlled PC tools.

## Phase 18 — Basic Autonomous Task Orchestration Complete

Phase 18 contains a **basic implementation of all planned orchestration steps (104–109)**. The layer manages finite task graphs, dependencies, priorities, bounded retries and stop/resume control while keeping actual external actions behind explicit host-provided executors.

- Task queue: explicit task records with status, priority, dependencies and retry budget.
- Dependency scheduling: only tasks whose dependencies are completed become runnable.
- Priority selection: ready tasks are selected deterministically by priority and task ID.
- Bounded execution loop: a maximum step count prevents an unbounded control loop.
- Failure recovery: failed tasks can retry within their budget or be explicitly requeued.
- Brain integration: `TARABrain` can add, run, stop and resume orchestrated tasks.

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
Task Queue → Autonomous Orchestrator
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

Autonomous orchestration does not create permissions. A task can execute only through the callable supplied by the host application. The existing Tool Controller remains the permission boundary for PC actions. The orchestrator has bounded steps, explicit dependencies, retry limits, and stop/resume controls.

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

Phase 18 is complete at the **basic architecture level**. Future phases can deepen learned planning, richer dependency graphs, persistent scheduling, resource awareness, and stronger evaluation without removing these explicit control boundaries.
