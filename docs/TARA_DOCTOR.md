# TARA Doctor: Debugging and Command Development

TARA Doctor provides bounded diagnostics and a controlled way to propose new diagnostic commands.

## Commands

Run from the repository root:

    python -m src.doctor check
    python -m src.doctor status
    python -m src.doctor tests --timeout 300
    python -m src.doctor commands list
    python -m src.doctor commands suggest "pytest failing after recent change"
    python -m src.doctor commands scaffold profile_memory --description "Profile memory use"

## Capability status

`status` prints a JSON snapshot for the current checkout. It distinguishes source files, optional SDK installation, required configuration, adapter readiness, and whether a supported default-runtime entry point references the integration. Missing Ollama and public-API source files are reported as unavailable on this branch, not deleted or retired. The report never prints environment-variable values and makes no network requests. A source-level adapter can be ready without being enabled in the default runtime.

## What it checks

- Python syntax and required core files through the existing static repository auditor.
- Duplicate full-suite pytest workflows and non-deterministic built-in hash use.
- Broad `except Exception: pass` blocks that may silently hide failures.
- TODO/FIXME/BUG markers that need triage.
- The fixed pytest suite, with timeout and captured output.

A clean report means only that the implemented checks found no issue; it does not prove the repository is bug-free.

## Teaching TARA to create commands

`commands suggest` maps a problem description to deterministic suggestions. `commands scaffold` creates a Python skeleton in `src/doctor_commands/`.

Generated commands are **not automatically registered or executed**. The skeleton intentionally raises `NotImplementedError` until a developer implements and tests it.

1. Describe the recurring bug or diagnostic need.
2. Ask Doctor for a suggested command.
3. Generate a scaffold with a valid command name.
4. Implement bounded, read-only checks first and add timeout/resource limits for expensive checks.
5. Add unit and failure tests, review permissions, then register explicitly after code review and CI pass.

Doctor never interprets problem descriptions as executable code, automatically applies speculative fixes, or runs model-generated shell commands. An eventual AI-assisted fixer should create a reviewable patch, test it in an isolated worktree/sandbox, show the diff, and require explicit approval before applying or merging.
