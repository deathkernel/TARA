# Memory

TARA uses separate but interacting memory systems rather than a single undifferentiated store.

## Canonical components

- `episodic.py`: detailed event/context traces with time, entities, events, facts, novelty and prediction error.
- `semantic.py`: generalized facts with confidence, evidence counts and provenance.
- `schema.py`: recurring event structures that become generalized schemas.
- `consolidation.py`: prioritized replay for surprise/importance/novelty.
- `system.py`: coordinates episodic encoding, semantic consolidation, schema learning and retrieval.
- `store.py`: storage protocol for future persistent backends.

## Scientific hypothesis

Recent work continues to support a distinction between episodic and semantic memory while also showing substantial interaction and representational overlap between them. citeturn804937search3

A 2025 framework proposes viewing episodic and semantic memory as parts of an online structure-learning problem: surprising episodes can be preserved in relatively raw form while semantic knowledge learns environmental regularities that compress experience. citeturn804937search1

A 2026 computational model explicitly explores hippocampo-neocortical interaction as a compression/retrieval system in which episodic traces can train a more general semantic model. TARA's implementation is a much smaller engineering analogue, not a claim of biological equivalence. citeturn804937search9

## TARA memory loop

`experience -> episodic trace -> replay/consolidation -> semantic fact + schema -> retrieval -> working memory`

Repeated structure is generalized into schemas; contradictory evidence reduces semantic confidence instead of silently overwriting the previous state.

## Development rule

Memory mechanisms must remain inspectable, source-aware and measurable. Persistent storage backends can be added later without changing the cognitive contracts.
