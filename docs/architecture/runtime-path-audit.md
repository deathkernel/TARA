# TARA Runtime Path Audit

**Reviewed:** 2026-10-15  
**Scope:** `main` plus the current unmerged PR stack (#23–#26). This document distinguishes the baseline from proposed branch behavior.

## 1. Baseline on `main`

- The inspected `main` tree does not contain `src/__main__.py`, `src/ollama_provider.py`, or `src/llm_provider_factory.py`.
- `src/brain.py` imports `GroqProvider`; `requirements.txt` includes `groq`; `.env.example` includes Groq settings.
- The README describes Ollama chat and documents `python -m src`, so the README and the current `main` tree are not yet aligned.
- Therefore the documented chat command must not be reported as verified on `main` from the inspected evidence.

## 2. Proposed runtime path in the PR stack

The current stacked branch `feat/chat-public-api-commands` contains the following files:

- `src/__main__.py`: CLI entrypoint and chat loop.
- `src/llm_provider_factory.py`: provider selection from `TARA_LLM_BACKEND`.
- `src/ollama_provider.py`: HTTP adapter to the local Ollama `/api/chat` endpoint.
- `tests/test_tara_cli.py`, `tests/test_llm_provider_factory.py`, and `tests/test_ollama_provider.py`: isolated tests for CLI routing and provider behavior.
- `src/public_api_tools.py`: bounded read-only external API clients, surfaced by the CLI in PR #26.

Proposed startup sequence:

1. `main()` defaults `TARA_LLM_BACKEND` to `ollama`.
2. `provider_from_env()` selects `OllamaProvider` for `ollama`/`llama`, or returns no provider for `none`/empty/local.
3. `OllamaProvider.from_env()` reads `TARA_LLAMA_MODEL`, `OLLAMA_HOST`, `TARA_LLM_TIMEOUT_SECONDS`, `TARA_POWER_MODE`, and the `TARA_OLLAMA_*` resource settings.
4. `TARABrain` receives the provider and `respond()` calls its `complete()` method with TARA-built context and system guidance.
5. If the provider call fails, the CLI brain's fallback `generate()` raises an explicit no-checkpoint-fallback error; the CLI catches provider/runtime errors and reports them rather than silently claiming success.
6. Explicit `/api ...` commands call `PublicAPITools` directly. The CLI's summaries label remote data untrusted and do not pass it to the language model.

## 3. Provider configuration in the proposed branch

- `TARA_LLM_BACKEND=ollama` selects local Ollama.
- `TARA_LLAMA_MODEL` defaults to `llama3.2:3b`.
- `OLLAMA_HOST` defaults to `http://127.0.0.1:11434`.
- `TARA_POWER_MODE` supports `eco`, `balanced`, and `performance`; `eco` is the default.
- Additional controls include `TARA_LLM_TIMEOUT_SECONDS`, `TARA_OLLAMA_NUM_CTX`, `TARA_OLLAMA_NUM_THREADS`, `TARA_OLLAMA_NUM_BATCH`, and `TARA_OLLAMA_KEEP_ALIVE`.
- Provider construction does not start Ollama or download a model. The local service and model must be installed separately.
- A configuration-order defect was found during this pass: the factory originally read environment variables before loading `.env`. The current PR #26 branch now calls `load_dotenv()` before provider selection and adds a regression test that selects Ollama from a temporary `.env` file.

## 4. PR-stack dependency and history review

| PR | Proposed change | Dependency / review note |
|---|---|---|
| #23 | Remove Groq provider integration | Base is `fix/teacher-curation-empty-output`, not `main`; review its inherited Ollama runtime before treating it as a standalone patch. |
| #24 | Add safe read-only public API tools | Based on PR #23's branch. |
| #25 | Retire dataset/training/checkpoint and persistent-memory workflows | Based on PR #24; removes many files, so historical docs must be archived rather than silently lost. |
| #26 | Expose public API commands in chat | Based on PR #25; previously deleted `README_COMMITS.md`. A historical phase/commit summary has now been restored on this feature branch and removed legacy workflow docs have been copied to `docs/archive/legacy-training/`. |

All four PRs were open and unmerged at the time of this audit. The files above describe proposed branch state, not shipped `main` behavior.

## 5. Remaining verification work

- Run the complete Python tests against the final stacked branch after history-preservation edits and the new dotenv regression test.
- Verify `python -m src` starts with Ollama unavailable, reports a useful error, and does not download models or start background services.
- Test one real Ollama request only in an environment where Ollama and the selected model are installed; mock-based provider tests do not prove a live server works.
- Test the public API client using its unit tests and verify redirect rejection, response bounds, timeouts, and source-link rendering.
- Check the Docker/Compose validation jobs; they have not been proven by documentation alone.
- Reconcile the README quickstart and environment example with the actual provider settings on the final branch.
