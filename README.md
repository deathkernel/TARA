# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a research-first neural-network project built from mathematical and implementation fundamentals. The goal is a small, inspectable artificial reasoning system that can connect a neural language core with perception, memory, reasoning, planning, reflection, controlled PC tools, autonomous tasks, scheduling and an explicit world model.

## Phase 20 — Basic World Model and Event Routing Complete

Phase 20 contains a **basic implementation of all planned world-model steps (116–121)**. TARA can maintain a bounded explicit world state, record normalized events, assemble read-only context for reasoning, route selected events to explicit host actions, and expose the world model through the brain.

- World state: explicit key/value state with bounded event history.
- Event stream: normalized timestamped `WorldEvent` records.
- Context assembly: bounded recent events plus a state snapshot.
- Event routing: event types can trigger only explicitly registered host callables.
- Brain integration: observations update the world model and world events can be observed directly.
- Prompt context: world state can be converted into a compact reasoning context.

## Current architecture

```text
Input / Perception
       ↓
Working Memory → Long-Term Memory
       ↓
World Model → Event Router → Context
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

The world model stores state and routes events, but does not grant permissions. Event handlers are explicit host-provided callables. Scheduled tasks still execute only through the bounded host executor, and PC actions remain behind the Tool Controller permission boundary.

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

### Phase 20 — World Model / Event Routing

116. ✅ Explicit world state
117. ✅ Bounded event history
118. ✅ Reasoning context assembly
119. ✅ Event routing to explicit host handlers
120. ✅ Brain integration
121. ✅ World-state prompt context

Phase 20 is complete at the **basic architecture level**. Future phases can deepen perception adapters, persistent world-state storage, richer event schemas, temporal reasoning, and learned environment modeling without removing the explicit control boundaries.
