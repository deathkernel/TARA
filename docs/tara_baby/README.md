# TARA Baby

TARA Baby is the canonical vNext program for building TARA as a conversational, emotionally aware, scientifically grounded personal assistant.

The project is organized around four boundaries:

1. Neural core — language prediction and learned representations.
2. Mind systems — attention, working memory, long-term memory, emotion/appraisal, reasoning, planning, reflection and world modeling.
3. Action systems — Windows, browser, files, voice and explicitly permissioned tools.
4. Scientific curriculum — language, conversation, mathematics, physics and the natural sciences.

TARA Baby does not attempt to reproduce the human brain neuron-for-neuron. The engineering target is a compact computational architecture inspired by useful cognitive functions, with explicit measurements and reproducible experiments.

Canonical architecture: docs/tara_baby/ARCHITECTURE.md
Canonical curriculum: docs/tara_baby/CURRICULUM.md
Dataset policy: data/curriculum/tara_baby_v1.json

## Repository rule

New vNext code belongs under src/tara_mind/.
New training entry points belong under scripts/.
Dataset manifests belong under data/curriculum/.
Design and research documents belong under docs/tara_baby/.

The existing large src/ surface is treated as legacy/compatibility code until each module is deliberately migrated and tested. We do not add more unrelated modules to the repository root.
