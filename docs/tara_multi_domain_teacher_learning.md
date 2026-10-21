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
