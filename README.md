# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a research-first neural-network project built from mathematical and implementation fundamentals. The goal is a highly capable, inspectable artificial reasoning system that can connect a neural language core with perception, memory, reasoning, planning, reflection, controlled tools, autonomous tasks, scheduling, temporal context and an explicit world model.

## Phase 21 — Perception and Temporal Awareness Complete

Phase 21 implements steps **122–127** as an integrated perception pipeline:

- **Perception adapters:** existing document, system-state and screen-summary adapters now feed a canonical observation layer.
- **Input normalization:** heterogeneous strings/mappings become deterministic `NormalizedObservation` records.
- **Confidence-aware observations:** each observation carries bounded confidence, source, kind, timestamp and stable identity.
- **Temporal ordering:** observations are deduplicated and ordered deterministically by timestamp and identity.
- **Temporal reasoning context:** bounded temporal windows expose pairwise time relations and confidence filtering for downstream reasoning.
- **Brain integration:** `TARABrain` now maintains temporal perception and can assemble memory + world + temporal context before generation.

## Current architecture

```text
Input / Perception Adapters
       ↓
Normalization → Confidence → Temporal Ordering
       ↓
Working Memory → Long-Term Memory
       ↓
World Model → Event Router → Temporal Context
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

Perception and temporal context are data-processing layers; they do not grant permissions or execute host actions. The world model stores state and routes events only to explicit host-provided callables. Scheduled tasks execute through bounded host executors, and PC actions remain behind the Tool Controller permission boundary.

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

### Phase 21 — Perception / Temporal Awareness

122. ✅ Perception adapters
123. ✅ Input normalization
124. ✅ Confidence-aware observations
125. ✅ Temporal event ordering
126. ✅ Temporal context for reasoning
127. ✅ Brain integration

Phase 21 is complete at an **advanced implementation level**. The temporal layer is model-agnostic, bounded and deterministic, so future learned perception models can plug into the same observation contract.
