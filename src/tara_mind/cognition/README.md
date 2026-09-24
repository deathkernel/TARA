# Cognition

Attention, working memory, reasoning, executive control and appraisal.


## Canonical components

- `attention.py`: bounded, interpretable selective-attention hypothesis grounded in cognitive/neuroscience literature.
- `working_memory.py`: bounded active-state store motivated by working-memory capacity, attention, and interference research.
- `state.py`: inspectable working cognitive state and affect/appraisal signals.
- `loop.py`: orchestration shell for the observe-to-memory cognitive cycle.

New cognition modules must document a scientific motivation, computational hypothesis, measurable evaluation, and failure modes.

- `prediction.py`: explicit prediction, precision-weighted error, and surprise.
- `predictive_map.py`: successor-style future-state representation for learned transitions.
- `world_model.py`: online state-transition model connecting prediction, observation, surprise, and update.

These components are deliberately separate from the Transformer so the scientific hypotheses can be evaluated independently before being fused into training or inference.