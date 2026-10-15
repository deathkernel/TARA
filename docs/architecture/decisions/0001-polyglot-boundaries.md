# ADR 0001: Explicit Polyglot Boundaries

- **Status:** Proposed
- **Date:** 2026-10-15
- **Decision owners:** TARA maintainers

## Context

TARA currently contains a large Python codebase and an independently tested Rust tokenizer crate. The repository does not yet document a stable Python↔Rust runtime contract. Adding languages without defined responsibilities would increase operational complexity without demonstrated benefit.

## Decision

1. Keep Python as the primary orchestration and integration language for the current architecture.
2. Keep Rust components independently buildable and tested. Integrate them into the Python runtime only after a concrete requirement, API contract, benchmark, and failure-handling plan are reviewed.
3. Do not add TypeScript, Go, or a database dependency until a UI, service boundary, or persistence requirement exists.
4. Use Docker/Compose to make validation reproducible; do not claim an application service is runnable until a real entrypoint is implemented and tested.
5. Preserve historical training/data work and progress records until a separate retirement change has been reviewed.

## Consequences

- CI and local development can validate Python and Rust separately.
- Cross-language boundaries remain explicit rather than relying on undocumented subprocess or FFI behavior.
- Future components must justify their maintenance and security cost with evidence.
