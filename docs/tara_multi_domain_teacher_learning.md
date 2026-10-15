# TARA multi-domain teacher learning

TARA can use the local Ollama model as a teacher to create candidate training data across several capability areas, not only conversation memory. This is a data-generation stage in a larger teacher/student workflow.

## Domains in v1

- Mathematics
- Logical reasoning
- Python coding
- Algorithms and complexity
- Language and communication
- General scientific knowledge
- Planning
- Tool use and verification
- Safety and uncertainty

## Generate a small local sample

Ensure Ollama is running and the model is installed:

```bat
ollama list
python -m src.teacher_distillation --per-domain 2 --output outputs/llama_teacher_smoke.jsonl
```

The smoke command requests 18 candidate examples (2 for each of 9 domains). Increase `--per-domain` gradually. The default output is `data/llama_teacher_v1.jsonl`; the command refuses to overwrite an existing file unless `--overwrite` is explicitly provided.

## Review, then train

Each JSONL record includes the instruction, answer, domain, teacher model, SHA-256 fingerprint, and `review_status=unverified_candidate`. The `text` field is compatible with TARA's current text-dataset training interface.

Teacher-generated answers can be wrong, repetitive, or poorly reasoned. The generator only performs basic structural checks and exact-record deduplication; it does not prove factual or code correctness. Review/filter examples and evaluate on a separate held-out set before training.

After review, run TARA's existing explicit training workflow against a curated dataset and a separate candidate checkpoint, for example:

```bat
python run_training_experiment.py --data data/llama_teacher_reviewed.jsonl --output checkpoints/llama_teacher_candidate.pt --steps 100
```

The reviewed dataset path above is an example: create it from checked examples first. This command trains TARA's own model architecture; it does not change Llama's weights. To claim improvement, compare baseline and candidate on held-out capability tests and keep the candidate only if it passes the existing evaluation gate. Do not overwrite user checkpoints, personal conversation data, or experiment outputs.

## What this is and is not

- This pipeline uses Llama-generated answers as one teacher-data source; it does not access Llama's raw pretraining corpus.
- It complements public datasets, curated examples, verified tool outcomes, and user-approved conversation memory.
- Memory updates improve retrieved context. Model training updates a checkpoint. These are separate mechanisms.
- Large-scale teacher generation and training can consume substantial time, RAM, disk, and CPU/GPU resources; start small.


## Batch C: audit, review, and held-out evaluation

### 1. Audit candidate data (read-only)

```bat
python -m src.teacher_dataset_audit outputs\teacher_candidates.jsonl --report outputs\teacher_audit.json
```

The audit validates required fields, supported domain, non-empty content, canonical
text rendering, teacher provenance, review state, per-record SHA-256 fingerprint,
malformed JSON, duplicate fingerprints, and duplicate instruction/answer pairs. It
does not judge truth or executable-code correctness. Existing report paths are never
overwritten. Exit code 2 means an error was found (or a warning when
`--fail-on-warnings` is supplied).

### 2. Human review queue

```bat
python -m src.teacher_review export outputs\teacher_candidates.jsonl outputs\teacher_review.csv
```

Open the CSV in a spreadsheet editor. Check every answer against independent reasoning,
tests, or a trusted source; fill `decision` with exactly `approve` or `reject`, and
enter a reviewer identifier. Do not mark answers approved merely because they look fluent.

```bat
python -m src.teacher_review import outputs\teacher_candidates.jsonl outputs\teacher_review.csv data\teacher_approved.jsonl
```

Import requires one explicit decision per candidate and rejects changed answer text,
unknown/duplicate fingerprints, missing reviewers, invalid decisions, or pre-existing
output files. Only approved rows are emitted. The source candidate JSONL and review CSV
are read-only inputs to import.

### 3. Held-out checkpoint comparison

```bat
python -m src.teacher_heldout_eval checkpoints\baseline.pt checkpoints\phase_c_candidate.pt outputs\phase_c_heldout.json
```

The evaluator runs the same deterministic capability suite against both checkpoints and
records overall scores, per-category deltas, suite fingerprint, evaluation fingerprints,
and the strict regression-gate decision. Exit code 2 means the candidate was not accepted.
This fixed suite must be supplemented with task-specific tests that were not used in
teacher generation or training.

### Batch C handoff

- Candidate generation, audit, review, and held-out comparison are implemented in code.
- Generated answers are still untrusted until independently reviewed.
- No checkpoint is promoted, replaced, or deleted by these commands.
- CI, Windows focused tests, and full Python/Rust validation must pass before integration.


## Current handoff — 15 October 2026

- Branch: `feature/tara-phase-c-teacher-data-quality`
- Draft PR #16: https://github.com/deathkernel/TARA/pull/16 (stacked on Batch B PR #15).
- CI was queued at last check; exact-head CI must pass before this batch is considered validated:
  - https://github.com/deathkernel/TARA/actions/runs/38695774939
  - https://github.com/deathkernel/TARA/actions/runs/38695774821
- Next steps: inspect exact-head CI results, resolve any reproduced failures, run focused and full Python/Rust tests on Windows using a separate worktree, inspect diffs, and obtain explicit approval before merging.
- No local Windows suite or live teacher/model training was executed during remote repository edits.


### Batch C handoff update — 15 October 2026

PR #16 remains a draft stacked on Batch B PR #15: https://github.com/deathkernel/TARA/pull/16.
The branch is rebased onto the latest Batch B fix that runs experiment scripts as repository
modules, preserving that fix and its CI command. Exact-head CI, Windows tests, and the full Python/
Rust suites still need to pass before integration. No merge is performed without explicit approval.


## Batch D — approval-gated model learning

Batch D connects the reviewed teacher corpus to an explicit TARA candidate-training run.
It does not train Llama, silently train during chat, or promote checkpoints automatically.

```mermaid
flowchart TD
    A[Teacher candidate JSONL] --> B[Batch C structural and provenance audit]
    B --> C[Human review CSV]
    C --> D[Approved-only curated JSONL]
    D --> E[Batch D strict approval gate]
    E --> F[Train separate TARA candidate checkpoint]
    F --> G[Evaluate baseline and candidate on same held-out capability suite]
    G --> H{Strict improvement and no regression?}
    H -->|Yes| I[Evidence report says accepted]
    H -->|No| J[Evidence report says rejected]
    I --> K[Manual review; no automatic promotion]
    J --> K
```

### Run on Windows CMD

Use paths that do not already exist. Keep the baseline checkpoint read-only and choose a new candidate path.

```bat
python -m src.teacher_dataset_audit data\\teacher_approved.jsonl --report outputs\\teacher_approved_audit.json
python -m src.teacher_training data\\teacher_approved.jsonl checkpoints\\baseline.pt checkpoints\\batch_d_candidate.pt outputs\\batch_d_evidence.json --steps 100
```

The first command is optional preflight; the training command repeats the audit itself and blocks unless every row has `review_status=approved`, a review decision of `approve`, and a non-empty reviewer. Duplicate or structurally invalid data blocks training. Existing candidate/report paths are never overwritten. The candidate is a separate artifact even when the held-out gate rejects it; rejection never deletes it or promotes it.

The report records the dataset fingerprint and per-domain counts, training manifest, baseline/candidate evaluation, category deltas, decision, and report fingerprint. The gate requires a strict overall improvement and no disallowed category regression under the existing evaluator. This fixed capability suite is a regression signal only, not proof of general intelligence or correctness on every teacher example.

### Scope and validation status

- This is an explicit local training command; no training happens during chat/runtime startup.
- It trains TARA's own small language model, not Llama weights.
- Start with a small `--steps` value and monitor RAM, CPU/GPU, disk, and runtime.
- A generated candidate is not automatically the production checkpoint.
- Remote code edits do not establish that tests passed. Exact-head CI and Windows/full-suite validation must be checked separately.


## Batch E — end-to-end teacher pipeline integration

Batch E connects the earlier generation, audit, human-review, curation, training, and held-out evaluation pieces into one recoverable command-line workflow. It intentionally pauses for a human to review the CSV; it never auto-approves teacher answers.

```mermaid
flowchart TD
    A[Local Ollama teacher] --> B[prepare: generate candidates]
    B --> C[Structural and provenance audit]
    C --> D[Export review CSV and manifest]
    D --> E[Human verifies every answer]
    E --> F[curate: import explicit approve/reject]
    F --> G[Approved-only JSONL and curation manifest]
    G --> H[train: train separate TARA candidate]
    H --> I[Evaluate baseline and candidate]
    I --> J{Strict improvement without regression?}
    J -->|Yes| K[Record accepted evidence]
    J -->|No| L[Record rejected evidence]
    K --> M[Manual review only; never auto-promote]
    L --> M
```

### Stage 1 — generate, audit, and export the review queue

Configure the local provider and verify Ollama/model availability first:

```bat
ollama list
set TARA_LLM_BACKEND=ollama
python -m src.teacher_pipeline prepare --dataset outputs\batch_e_candidates.jsonl --audit-report outputs\batch_e_audit.json --review-csv outputs\batch_e_review.csv --manifest outputs\batch_e_prepare_manifest.json --per-domain 2
```

The example requests 18 candidate examples (two per each of the nine domains). Start small. Each output path must be new and all artifact paths must be distinct. The command refuses to overwrite the dataset, generator temporary file, audit report, review CSV, or manifest. A clean audit establishes schema/provenance consistency only—not factual correctness.

### Stage 2 — human review and curation

Open `outputs\batch_e_review.csv` in a spreadsheet editor. Independently verify each answer, fill every `decision` with exactly `approve` or `reject`, and provide a reviewer identifier. Never approve an answer solely because it sounds fluent.

```bat
python -m src.teacher_pipeline curate --dataset outputs\batch_e_candidates.jsonl --review-csv outputs\batch_e_review.csv --approved-dataset data\batch_e_approved.jsonl --manifest outputs\batch_e_curation_manifest.json
```

Curation refuses incomplete decisions, changed content, unknown/duplicate fingerprints, missing reviewer attribution, duplicate records, and existing output paths. The output JSONL contains only explicitly approved rows. Candidate JSONL and review CSV are inputs and are not edited by this command.

### Stage 3 — train a separate candidate and evaluate it

Use an existing baseline checkpoint, but choose a new candidate checkpoint and evidence-report path:

```bat
python -m src.teacher_pipeline train --dataset data\batch_e_approved.jsonl --baseline checkpoints\baseline.pt --candidate checkpoints\batch_e_candidate.pt --report outputs\batch_e_evidence.json --steps 100
```

The training stage repeats the approval/provenance checks, trains TARA's own small model (not Llama weights), and evaluates baseline and candidate on the same held-out capability suite. The report includes dataset/training fingerprints, category deltas, and the evidence-gate decision. A rejected candidate is retained for inspection; no checkpoint is promoted or deleted automatically.

### Recovery model

Stages are separate on purpose: the CSV review is a human boundary, and each stage emits a new artifact. If a stage fails, inspect the last successful manifest/report and fix the issue; do not delete or overwrite artifacts blindly. Use new output paths for reruns. If training fails after curation, the approved JSONL remains available and can be passed to `python -m src.teacher_training` directly with fresh candidate/report paths.

### Batch E handoff — 15 October 2026

- Branch: `feature/tara-phase-e-teacher-pipeline-integration`, stacked on Batch D `feature/tara-phase-d-model-learning`.
- Added `src/teacher_pipeline.py` with `prepare`, `curate`, and `train` stages; tests are in `tests/test_teacher_pipeline.py`.
- CI workflow adds Batch E branch/path triggers and the new focused test module.
- Exact-head CI and local Windows/full Python/Rust validation must be checked; remote edits alone are not test evidence.
- No real Ollama generation or checkpoint training was executed while editing remotely.
- Next: inspect CI and fix reproduced failures; run focused tests and full suites in a separate worktree; review dependency-stack diffs; ask for explicit merge approval before integration.
- Preserve local `checkpoints/`, `outputs/`, conversation JSONL/manifests, and `test_voice.py`. Do not run destructive cleanup or merge without explicit approval.


## Batch F — final runtime integration and release gate

Batch F closes this implementation roadmap with a deterministic integration check across
TARA's cognitive runtime, persistent memory, explicit provider selection, identity guardrails,
and learning orchestration. It adds regression coverage and a dedicated CI workflow; it does
not claim that a new foundation model has been trained or that live Ollama inference has passed.

### Integrated runtime path

```mermaid
flowchart TD
    A[Explicit provider configuration] --> B[TARABrain]
    B --> C[TARACore health and bounded orchestration]
    C --> D[Plan and execute deterministic test actions]
    D --> E[Record verified learning experience]
    E --> F[Persist and reload user memory]
    F --> G[Write runtime manifest and fingerprint]
    G --> H[Release regression gate]
    H --> I[Full Python test suite]
    H --> J[Rust tokenizer tests]
    H --> K[Teacher pipeline CLI surface]
    I --> L{All checks pass?}
    J --> L
    K --> L
    L -->|Yes| M[Review exact commit and artifacts]
    L -->|No| N[Fix evidenced failure and rerun]
    M --> O[Human approval before merge]
```

### What Batch F adds

- `tests/test_phase_f_integration.py` composes the existing brain/core, explicit provider
  factory, persistent memory round-trip, runtime manifest, deterministic action cycle, and
  verified-learning API in one regression path.
- `.github/workflows/phase-f-release-gate.yml` runs the full Python suite, checks the teacher
  pipeline CLI help without generating teacher data, and runs the Rust tokenizer tests.
- The integration test uses a dummy model and deterministic actions. It does not contact
  Ollama, call remote model APIs, generate teacher answers, train checkpoints, or promote a model.
- Model/provider selection stays explicit. The default provider remains unchanged unless
  `TARA_LLM_BACKEND` is configured.
- Runtime memory, model checkpoints, teacher datasets, and generated evidence remain separate
  artifacts; the release gate does not delete or overwrite user data.

### Windows CMD validation

Run these from a separate validation worktree so existing local artifacts and uncommitted work
in the primary checkout remain untouched:

```bat
python -m pytest tests/test_phase_f_integration.py tests/test_tara_core.py tests/test_memory.py tests/test_llm_provider_factory.py -q
python -m pytest -q
python -m src.teacher_pipeline --help
cargo test --manifest-path rust-tokenizer/Cargo.toml
```

The commands above are instructions, not proof that they have already passed. Confirm the exact
Batch F commit's GitHub Actions runs and run the Python/Rust checks locally before considering
the stack validated.

### Batch F handoff — 15 October 2026

- Branch: `feature/tara-phase-f-release-integration`, stacked on Batch E.
- Draft PR #19: https://github.com/deathkernel/TARA/pull/19, targeting `feature/tara-phase-e-teacher-pipeline-integration`.
- New files: `tests/test_phase_f_integration.py` and
  `.github/workflows/phase-f-release-gate.yml`.
- Validation remains pending until exact-head CI and separate-worktree Windows tests have been
  inspected. No live teacher generation, real training, or Ollama inference is claimed.
- Do not merge or close stacked PRs without explicit approval. Preserve local `checkpoints/`,
  `outputs/`, conversation JSONL/manifests, and `test_voice.py`; do not run destructive cleanup.
