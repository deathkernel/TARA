# TARA Current-State Architecture Audit

**Baseline:** `main` at `19f4eb239ef75f38bb09a89a19b31fe45d8eb86e`  
**Audit type:** repository tree and selected source/config inspection  
**Status:** initial audit; not a claim that every implementation path has been executed.

## Evidence-backed inventory

- Repository tree contains 281 entries at the audit baseline.
- Python code is concentrated in `src/`, with many modules at one flat level, including cognitive/runtime code, model/training research, provider integration, safety/tools, and evaluation.
- Rust exists in `rust-tokenizer/` as a Cargo package with `src/lib.rs` and `src/main.rs`. Its manifest describes a dependency-free educational BPE tokenizer. CI runs `cargo test --all-targets` from that directory.
- Tests are concentrated in `tests/`; the repository has a large number of module-level tests.
- CI currently has Python and Rust jobs in `.github/workflows/tests.yml`. The Python job installs `requirements.txt`, runs `python -m src.repository_audit .`, runs pytest, and also executes XOR/tiny-language-model smoke experiments.
- `requirements.txt` currently declares `numpy`, `scikit-learn`, `pyttsx3`, `datasets`, `pytest`, `torch`, `pyarrow`, `groq`, and `elevenlabs`.
- `.env.example` currently documents Groq and ElevenLabs configuration. This does not match the README's claim that Ollama is the current local setup; provider/runtime truth must be reconciled against the active implementation and PR stack.
- `src/brain.py` imports `GroqProvider`; `src/groq_provider.py` and its tests exist on main.
- `src/tara_core.py` describes a top-level cognitive runtime, but it is not itself a CLI entrypoint.
- Root-level training/dataset scripts and experiments remain on main. The README labels these as historical for normal Ollama chat, but they have not been physically retired from this baseline.
- Root `Dockerfile`, Compose manifest, root `pyproject.toml`, and `src/__main__.py` were not found in the inspected tree. Therefore the documented `python -m src` command has no matching `src/__main__.py` in this baseline and should be validated/fixed rather than assumed executable.
- Existing GitHub Actions workflows are present; do not replace them wholesale without preserving their purpose and history.

## Confirmed gaps

1. **Runtime truth gap:** README, environment example, dependency list, and the main-branch provider implementation do not currently tell one consistent story about Ollama vs Groq.
2. **Packaging/entrypoint gap:** no root packaging manifest or `src/__main__.py` was found in the tree.
3. **Container gap:** no Dockerfile or Compose configuration was found at the root or under the inspected paths.
4. **Boundary gap:** the tree mixes runtime modules with historical training/learning experiments and has a large flat `src/` surface.
5. **Polyglot integration gap:** Rust exists and is CI-tested, but the inspected evidence does not show a defined Python↔Rust runtime boundary. Do not claim the Rust tokenizer is used by the chat path without integration evidence.
6. **CI responsibility gap:** the primary Python CI job combines ordinary tests with training/experiment smoke scripts, which makes the runtime validation path harder to distinguish from historical research validation.

## Migration guardrails

- Preserve the phase ledger, `README_COMMITS.md`, and historical engineering evidence.
- Do not delete old training/data files merely to make the tree look cleaner. First isolate their ownership and dependencies; retire them in a dedicated reviewed change.
- Do not move modules in bulk until import graphs and tests have been mapped.
- Keep Python↔Rust boundaries explicit and versioned; add FFI only when a concrete runtime use case and benchmark justify it.
- Keep secrets out of images and source control. Use environment injection, bounded permissions, health checks, and explicit network/service dependencies.
- Keep all new changes on a feature branch and do not merge without explicit approval.

## Next audit pass

- Inspect imports and runtime construction across `brain.py`, `basic_brain.py`, `agent.py`, `tara_core.py`, provider adapters, and the actual chat entrypoint.
- Reconcile open PRs before deciding whether Groq, Ollama, training, and persistence code are current or being retired.
- Add architecture decision records and establish baseline test results before file moves.
- Establish containerized validation first; introduce an application container only once a real, tested application entrypoint exists.
