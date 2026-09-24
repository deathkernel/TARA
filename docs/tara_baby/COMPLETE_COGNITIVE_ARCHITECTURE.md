# TARA Baby Complete Cognitive Architecture

## Research-driven design

The architecture is now organized around a closed cognitive-control loop rather than a collection of unrelated AI features. The neural Transformer remains the learned language substrate; explicit cognitive mechanisms handle state, prediction, memory, evaluation and action control.

Current research motivates several of these separations. Goal-directed behavior involves hierarchical prefrontal-striatal control; metacognition uses confidence/uncertainty signals; memory research emphasizes interaction between episodic traces, semantic abstraction, replay and schema learning. citeturn210531search6turn210531search0turn210531search4turn210531search17

## Closed loop

`observe -> attention -> working memory -> world state -> prediction -> appraisal -> reasoning -> metacognition -> planning -> executive gate -> action -> outcome -> reflection -> replay/consolidation -> memory`

## Canonical modules

### Neural substrate
- `src/tara_mind/core/tokenizer.py`
- `src/tara_mind/core/transformer.py`

### Cognition
- attention
- bounded working memory
- world model
- prediction error
- successor-style predictive map
- appraisal/affect control state
- scientific reasoning
- metacognitive monitoring
- model-based planning
- executive goal/action control
- reflection

### Memory
- episodic memory
- semantic memory
- schema learning
- replay/consolidation
- persistent SQLite storage
- unified memory system

### Continual learning
- replay new experiences
- consolidate semantic facts and schemas
- retain provenance and uncertainty
- evaluate before changing neural weights

### Automation
Tools remain behind an explicit allowlist and confirmation gate. The learned model does not receive unrestricted operating-system control.

## Scientific constraints

1. Each module must have a scientific motivation and a falsifiable computational hypothesis.
2. Brain-region analogies are functional, never literal one-to-one mappings.
3. Human subjective experience is not inferred from implementation variables.
4. Training loss is not treated as proof of cognition.
5. New mechanisms require behavioral evaluation before integration into training.
6. Coding data is not part of the intended training curriculum.

## Next engineering stage

The architecture layer is complete enough to begin integration experiments. The remaining work is empirical: run the unit suite locally, benchmark each mechanism, connect the cognitive controller to the neural model, and train/evaluate on the staged science-and-conversation curriculum without pretending that benchmark scores equal a human brain.
