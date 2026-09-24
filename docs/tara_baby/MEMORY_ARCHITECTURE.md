# TARA Baby Memory Architecture

## Research-derived direction

TARA uses a dual-timescale memory design:
- episodic memory keeps event-specific detail;
- semantic memory stores generalized knowledge;
- schema learning extracts recurring structure;
- replay gives surprising and important experiences more consolidation opportunities.

This is inspired by current work on episodic/semantic interaction, adaptive compression, hippocampal replay and schema learning. Research also shows overlap between episodic and semantic retrieval, so TARA does not treat them as isolated biological boxes. citeturn804937search1turn804937search3turn804937search9

## Computational flow

experience
  ↓
episodic encoding
  ↓
prediction error / novelty / importance
  ↓
priority replay
  ↓
repeated structure?
  ↓
schema formation
  ↓
semantic abstraction
  ↓
consolidation
  ↓
retrieval to working memory

## Why episodic memory stays detailed

A 2025 framework describes episodic memory as preserving surprising experiences in relatively raw form while semantic knowledge learns regularities that make future encoding more efficient. TARA therefore does not immediately replace an episode with a summary. citeturn804937search1

## Why schemas exist separately

Schemas are generalized structures about typical event sequences. A 2025 neuroscience perspective links schema learning to prediction errors, hierarchical learning and simplified representations of the environment. TARA's first implementation is intentionally simpler: repeated context-plus-sequence patterns become candidate schemas. citeturn804937search4

## Biological caution

Hippocampal circuits are strongly involved in episodic memory, but modern work emphasizes distributed interactions with neocortical and other networks. TARA treats the hippocampal/neocortical analogy as a functional design inspiration, not as a literal software-to-brain mapping. citeturn804937search0turn804937search10

## Implementation

- src/tara_mind/memory/episodic.py
- src/tara_mind/memory/semantic.py
- src/tara_mind/memory/schema.py
- src/tara_mind/memory/consolidation.py
- src/tara_mind/memory/system.py

## Future experiments

1. Test recall quality for exact episodes.
2. Test semantic generalization from repeated episodes.
3. Test schema formation under noisy sequences.
4. Measure whether replay priority improves adaptation without catastrophic overwriting.
5. Add durable storage only after the in-memory behavior is evaluated.
