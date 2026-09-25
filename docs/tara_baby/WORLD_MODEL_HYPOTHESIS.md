# World Model Hypothesis

TARA now separates the current world state from the language model and maintains a learned transition structure.

## Scientific grounding

Predictive-processing frameworks treat perception and cognition as involving hypotheses that are updated when observations disagree. Hippocampal predictive-map research provides another relevant computational idea: represent expected future states from learned transitions rather than only remembering the present state.

These are active research areas, so TARA treats both as testable computational hypotheses rather than as literal brain descriptions.

## Mechanism

`current state -> transition distribution -> predicted next state -> observation -> surprise -> update`

For an observed transition s -> s', TARA estimates:

`surprise = -log P(s' | s)`

A repeated transition raises its empirical probability; unexpected transitions produce larger surprise. Surprise can later feed attention, replay priority, reflection, and planning.

## Why this matters

This creates a bridge between perception and learning:

`experience -> prediction -> error/surprise -> memory priority -> model update`

The same mechanism can later support multi-step planning using the successor-style predictive map.
