# Phase 14 — Training ↔ Brain Integration

Phase 14 turns TARA's dataset/training foundation into a reproducible training path for the learned language core.

## Pipeline

```text
Dataset / API
    ↓
prepared JSONL
    ↓
TrainingPipeline
    ├── deterministic train/validation split
    ├── CharTokenizer
    ├── FastTinyLanguageModel
    ├── AdamW + gradient clipping
    └── validation loss
    ↓
self-contained .pt checkpoint
    ├── model state
    ├── optimizer state
    ├── model configuration
    ├── tokenizer vocabulary
    ├── training step
    ├── train/validation metrics
    └── dataset fingerprint
    ↓
TARABrain.from_checkpoint()
    ↓
observe → memory → generate → verify → recover
```

## Train

```bash
python train_algorithm_lm.py \
  --data data/algorithm_tasks.jsonl \
  --output checkpoints/algorithm_lm.pt \
  --steps 1000 \
  --validation-split 0.1
```

The command performs the actual training. Importing the training module does not train anything.

## Resume

```bash
python train_algorithm_lm.py \
  --data data/algorithm_tasks.jsonl \
  --output checkpoints/algorithm_lm.pt \
  --resume checkpoints/algorithm_lm.pt \
  --steps 1000
```

Resume requires the same dataset fingerprint and model configuration. This prevents silently pairing an optimizer/model state with incompatible training data or tensor shapes. The tokenizer is restored from the checkpoint rather than rebuilt, so vocabulary dimensions remain compatible.

## Brain handoff

```python
from src.brain import TARABrain

brain = TARABrain.from_checkpoint("checkpoints/algorithm_lm.pt")
response = brain.respond("Problem: sort a list")
print(response.text)
```

The brain loads the trained model through `src.model_runtime`, then exposes the same observation, memory, planning, generation, verification and recovery interface as an in-memory brain.

## What Phase 14 does not claim

- Code added to the repository is **not** the same thing as a trained model. Training must actually be run on the chosen dataset and hardware.
- Lower validation loss is an evaluation measurement, not proof of human-level reasoning.
- A generated algorithm is not automatically novel. Novelty still requires comparison against the relevant known algorithms and literature.
- The dataset fingerprint is a reproducibility/compatibility guard, not a license or quality approval.

## Checkpoint contract

A Phase 14 checkpoint contains:

- `format_version`
- `step`
- `model_state`
- `model_config`
- `tokenizer`
- `optimizer_state`
- `metrics`
- `dataset_fingerprint`
- complete training `config`

This gives later continual-learning and evaluation phases a stable handoff point.
