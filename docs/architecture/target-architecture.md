# TARA Target Architecture

This is a staged target, not a claim that every layer already exists. Implementation must follow repository evidence, tests, and approved PR direction.

## Runtime boundaries

```text
CLI / future HTTP API
        |
        v
Application bootstrap + validated configuration
        |
        v
TARA Core / cognitive orchestration
   |        |         |          |
   v        v         v          v
Reasoning  Memory   Tool policy  Observability
   |
   v
Inference provider interface
   |-------------------------------|
   v                               v
Ollama/local model              Optional provider adapters
(local HTTP API)                (only if explicitly supported)
   
Rust component boundary (separate, benchmark-driven)
   |
   +-- tokenizer library / CLI today
   +-- future native binding only if justified and tested
```

## Language policy

- **Python:** cognitive orchestration, provider adapters, developer tooling, integration tests, and research/evaluation modules that remain intentionally supported.
- **Rust:** performance-sensitive components with a clear contract. The existing `rust-tokenizer` stays an independently testable Cargo package until a measured integration path is approved.
- **TypeScript:** consider only if a browser UI or web control plane is in scope.
- **Go:** consider only if a standalone service/agent or operational requirement justifies a separate binary.
- **SQL:** add a schema/migrations layer only when a persistent database is selected.
- More languages are not automatically more advanced. Each additional runtime adds packaging, security, CI, observability, and maintenance cost.

## Proposed repository boundaries

```text
TARA/
├── src/                    # existing Python modules; migrate gradually
│   ├── app/                # bootstrap/configuration and application lifecycle
│   ├── core/               # orchestration and runtime contracts
│   ├── providers/          # inference-provider adapters
│   ├── cognition/          # reasoning, planning, perception, reflection
│   ├── memory/             # memory interfaces and policies
│   ├── tools/              # existing bounded tool interfaces and policy
│   ├── safety/             # authorization and action guardrails
│   └── observability/      # structured logs, health, tracing
├── rust-tokenizer/         # existing Rust crate; explicit independent boundary
├── tests/                  # preserve current suite; add architecture/integration tests
├── docs/architecture/      # audit, target design, ADRs
├── docs/roadmap/           # phase history and active roadmap
├── .github/workflows/      # CI matrix and separated research experiments
├── Dockerfile              # containerized Python validation/runtime base
├── compose.yaml             # local multi-service validation profile
├── pyproject.toml           # packaging/tool configuration after entrypoint decision
└── README.md                # current verified quickstart + progress ledger
```

Do not create empty folders just to match this drawing. Move a module only with an import map, compatibility plan, and passing regression tests.

## Infrastructure expectations

- Docker builds must be reproducible and run as a non-root user where possible.
- Compose must not bake API keys into image layers; secrets/config are injected at runtime.
- Health checks should report process/service readiness, not pretend a model is available when it is not.
- Local inference must declare whether it uses host Ollama or an Ollama service container. Avoid silently downloading large model weights during image builds.
- Python and Rust validation should be independently runnable, with a top-level Compose profile to run both.
- CI should separate deterministic unit tests from expensive research/training experiments.
- Add SBOM/image scanning and dependency auditing as a later hardening step, not as a false claim of current coverage.

## Delivery stages

1. **Audit + baseline:** document actual files, runtime entrypoint, providers, imports, CI, and PR dependencies.
2. **Containerized validation:** Python and Rust test containers; no false app-service claim until an app entrypoint exists.
3. **Runtime contract:** define one inference interface and reconcile README, config, provider implementation, and tests.
4. **Entrypoint/package:** add an actual CLI/API entrypoint and package metadata with tests.
5. **Incremental module boundaries:** migrate by domain in batches while preserving imports or adding compatibility shims.
6. **Optional services:** add HTTP API, UI, database, queue, or telemetry collector only when a real use case requires them.
7. **Hardening:** security scans, SBOM, resource limits, health checks, and deployment docs.
