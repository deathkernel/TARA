# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a research-first neural-network project built from mathematical and implementation fundamentals. The goal is a small, inspectable artificial reasoning system that can connect a neural language core with perception, memory, reasoning, planning and controlled PC tools.

## Phase 16 — Basic PC Tools Complete

Phase 16 now contains a **basic implementation of all planned tool steps (92–97)**. These are controlled interfaces, not unrestricted autonomous computer control.

- File tools: explicit allowed roots, bounded reads, confirmed destructive operations.
- Terminal tools: command allow-list, fixed working directory, timeout and output limits, `shell=False`.
- Application/browser tools: host-injected action registry; no implicit browser or application access.
- Keyboard/mouse tools: host-injected backend with basic coordinate/key validation.
- Tool controller: allow-list based selection, execution result capture and verification boundary.
- Emergency stop and rollback: stop blocks execution; reversible host actions can register rollback callbacks.

The actual host backend remains outside TARA's cognitive model. This keeps the basic architecture inspectable and avoids granting the model unrestricted OS access.

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
Reflection → Learning
       ↓
Emergency Stop / Rollback boundary
```

## Safety boundary

Phase 16 is intentionally basic. Tool access is explicit and host-controlled. File access requires configured roots; terminal execution requires an allow-listed executable; browser/application actions require registered host functions; keyboard/mouse operations require an injected backend. The tool controller can stop further execution and invoke registered rollback callbacks.

These controls are defense-in-depth and are not a claim of perfect isolation against hostile code. Truly untrusted code should use a separate OS/container/VM sandbox before production use.

## Roadmap

### Phase 16 — PC Automation / Tools

92. ✅ File tools
93. ✅ Terminal tools
94. ✅ Application/browser tools
95. ✅ Controlled keyboard/mouse interaction
96. ✅ Tool selection and execution verification
97. ✅ Emergency stop / rollback where possible

Phase 16 is complete at the **basic architecture level**. Future phases can deepen reliability, platform support, permissions, audit integration and sandboxing without changing these core boundaries.
