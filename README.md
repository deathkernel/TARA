# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a modular Python reasoning project. The repository contains components for language-model providers, cognitive/runtime experiments, memory, planning, tool interfaces, evaluation, and training. **A module existing in the repository does not mean it is wired into one supported end-user workflow.**

## What is verified on this branch

- **Groq provider adapter:** `src/groq_provider.py` implements an optional Groq-backed completion provider. It requires `GROQ_API_KEY` when instantiated without an injected client.
- **ElevenLabs text-to-speech adapter:** `src/elevenlabs_tts.py` provides optional synthesis/playback helpers. It requires credentials and the optional SDK.
- **Model/checkpoint utilities:** the repository contains separate training and checkpoint-related code; these are not required to use the optional hosted Groq provider.
- **Tests:** use the project's pytest suite to check code. A passing local test run and a passing CI run must be reported separately.

These are source-level capabilities, not a claim that a complete interactive chat application is currently available.

## Installation and validation

From the repository root:

```bash
python -m pip install -r requirements.txt
pytest
```

The full test suite may require platform-specific dependencies. Do not interpret installation or test commands as proof that every optional integration works end-to-end.

## Runtime status and configuration

There is no verified `src/__main__.py` entry point on the default branch at the time of this README audit. Therefore, this README intentionally does **not** advertise `python -m src` as a working chat launch command.

The checked-in `.env.example` documents these optional settings:

- Groq: `GROQ_API_KEY`, `TARA_GROQ_MODEL`, `TARA_GROQ_MAX_DAILY_REQUESTS`, `TARA_GROQ_MAX_DAILY_TOKENS`, and `TARA_GROQ_MAX_COMPLETION_TOKENS`.
- ElevenLabs: `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `ELEVENLABS_MODEL`, and `ELEVENLABS_OUTPUT_FORMAT`.

No `TARA_LLM_BACKEND` or Ollama configuration is documented as supported by the current checked-in `.env.example`. Do not assume that a provider is selected automatically; check the active code path and configure the provider explicitly.

## Proposed functionality

The following are **not documented as shipped on the default branch** unless their implementation PRs are merged and validated:

- Interactive Ollama chat and `python -m src` startup.
- Chat slash commands such as `/help` and `/exit`.
- Read-only `/api` commands for GitHub, Wikipedia, Open-Meteo weather, Hacker News, and Stack Exchange.
- TARA Doctor command scaffolding and diagnostics that are currently in review.

When these features are merged, update this section only after checking the implementation, tests, and exact default-branch revision.

## Architecture and safety principles

TARA's source tree includes components related to perception, reasoning, planning, memory, tool selection, reflection, world context, verification, and orchestration. Their degree of integration depends on the current code path; this README does not claim that every component participates in a single runtime loop.

1. **Inference is not training.** Calling a language model does not train it.
2. **Code is not evidence of integration.** Confirm entry points, call paths, tests, and actual behavior.
3. **Explicit permissions.** Model-generated text must not itself authorize external actions.
4. **Bounded integrations.** External requests should use validation, timeouts, response limits, and clear error handling.
5. **Evidence-based status.** Distinguish source-level implementation, automated tests, live integration tests, and measured performance.
6. **No false completion claims.** A historical phase label is not a production-readiness guarantee.

## Project history

The detailed phase ledger and historical notes are maintained in [`docs/PROJECT_HISTORY.md`](docs/PROJECT_HISTORY.md). Historical phases are context, not current setup instructions.

## Optional integrations

Keep API keys in local environment variables and never commit credentials. Optional provider integrations may require additional packages, valid credentials, network access, and live verification.

## Documentation maintenance

- Keep current setup instructions executable on the default branch.
- Document only environment variables read by current code.
- Label proposed or branch-only features as proposed.
- Move detailed historical phase notes out of the main README.
- Before claiming a feature is working, verify its entry point, implementation, tests, and CI on the exact revision.
