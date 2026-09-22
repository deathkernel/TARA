# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a research-first neural-network project built from mathematical and implementation fundamentals. The goal is a highly capable, inspectable artificial reasoning system that connects a neural language core with perception, memory, structured reasoning, planning, reflection, controlled tools, autonomous tasks, scheduling, temporal context and an explicit world model.

## Phase 22 — Structured Reasoning Complete

Phase 22 implements steps **128–134**:

- **Explicit reasoning state:** bounded, inspectable state for one reasoning goal.
- **Facts / assumptions separation:** claims are explicitly typed rather than mixed together.
- **Hypothesis generation:** candidate hypotheses receive stable identifiers and initial confidence.
- **Evidence tracking:** evidence is attached to claims with source, support/opposition and strength.
- **Contradiction detection:** explicit incompatible claims are surfaced instead of silently merged.
- **Reasoning verification:** required claims, contradictions and evidence confidence contribute to a deterministic verification result.
- **Brain integration:** `TARABrain.reason()` and `verify_reasoning()` expose the reasoning engine to the cognitive loop.

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
Structured Reasoning
   ├── Facts
   ├── Assumptions
   ├── Hypotheses
   ├── Evidence
   ├── Contradictions
   └── Verification
       ↓
Neural Language Core
       ↓
Planning
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

Structured reasoning records are inspectable data; they do not grant permissions or execute host actions. Tool use remains behind explicit controller boundaries and truly untrusted code should use a separate OS/container/VM sandbox.

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

### Phase 22 — Structured Reasoning

128. ✅ Explicit reasoning state
129. ✅ Facts / assumptions separation
130. ✅ Hypothesis generation
131. ✅ Evidence tracking
132. ✅ Contradiction detection
133. ✅ Reasoning verification
134. ✅ Brain integration

Phase 22 is complete at an **advanced implementation level**. The reasoning representation is intentionally auditable and model-agnostic; it is infrastructure for a future learned reasoner rather than a claim of human-level reasoning.
