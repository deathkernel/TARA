# Prediction, Predictive Maps, and Replay

TARA's next advancement combines three research-grounded mechanisms.

## 1. Prediction error

Predictive-processing accounts describe perception as involving hypotheses about incoming data and updates when evidence disagrees. The framework is useful here only where it can be made quantitative and testable; it is not treated as a settled theory of the whole brain. See the cited 2026 and 2025 reviews in the project notes.

TARA represents:

prediction -> observation -> signed error -> precision-weighted error -> surprise

The prediction engine can update a prediction toward new observations.

## 2. Hippocampal-style predictive maps

Recent computational neuroscience work models hippocampal predictive maps with successor representations that integrate spatial, contextual and motivational signals. A successor-style representation summarizes likely future states, which is useful for later planning and navigation.

TARA implements a discrete transition model and computes:

M = I + gamma*T*M

by fixed-point iteration. This gives an estimate of discounted future-state occupancy without assuming that software states correspond to neurons or place cells.

## 3. Replay and consolidation

Hippocampal replay is associated with memory consolidation, and recent work shows that replay during sleep participates in processing newly acquired memories. Awake replay is also observed.

TARA cannot reproduce biological sleep oscillations, so the engineering analogue is a replay buffer that prioritizes experiences using prediction error, importance and novelty, with diminishing returns from repeated replay.

## Why these three belong together

experience -> prediction -> error -> priority -> replay -> updated model

The goal is not to imitate a brain mechanistically, but to test whether these biological/computational principles improve retention, adaptation and planning.
