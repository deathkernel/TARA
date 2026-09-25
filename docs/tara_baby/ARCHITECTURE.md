# TARA Baby Architecture

## Goal

Build TARA Baby as a **science- and biology-grounded computational mind** for natural conversation, affect-aware interaction, scientific and mathematical reasoning, memory, planning, and controlled PC automation.

The neural language model is one subsystem. It is not the entire mind.

## Scientific stack

```
Biology / Neuroscience
        ↓
Cognitive hypotheses
        ↓
Mathematical representations
        ↓
Computational mechanisms
        ↓
Experiments + evaluations
        ↓
Revision
```

## Canonical mind loop

```
Environment / conversation
        ↓
Perception
        ↓
Attention
        ↓
Working memory
        ↓
World state / belief state
        ↓
Inference + prediction
        ↓
Planning
        ↓
Action selection
        ↓
Controlled tool/action interface
        ↓
Outcome observation
        ↓
Prediction-error / reflection
        ↓
Long-term memory + learning
        ↺
```

## Biological grounding

The architecture uses functional analogies to well-studied brain systems:

| Cognitive function | Biological motivation | Computational role |
| --- | --- | --- |
| Perception | hierarchical sensory processing | turn raw inputs into structured representations |
| Attention | selective/limited processing | allocate computation to relevant information |
| Working memory | limited active maintenance | hold current task state and intermediate results |
| Episodic memory | hippocampal–neocortical interactions | bind events, context and time for later retrieval |
| Executive control | prefrontal goal-directed behavior | maintain goals, resolve conflicts and choose actions |
| Affect/appraisal | interacting affective-cognitive systems | alter priorities, urgency, approach/avoidance and response style |

These are scientific design hypotheses, not claims that software modules correspond literally to individual brain regions. Working-memory capacity and prefrontal/hippocampal interactions are established research topics, while the exact mapping from neuroscience to artificial cognition is still open. citeturn524900search10turn524900search11turn524900search14

## Mathematical substrate

Canonical components should support:
- probability distributions and calibrated uncertainty
- vector/tensor representations
- optimization and gradient-based learning
- discrete symbolic structures where exactness matters
- state-space and dynamical models
- quantitative evaluation and error analysis

## Neural core

Current learned substrate:

- byte-level/subword tokenizer
- decoder-only Transformer
- rotary positional encoding
- RMSNorm
- causal self-attention
- SwiGLU feed-forward blocks
- tied language head
- AdamW training

Future changes must be justified by an explicit experiment and scientific hypothesis.

## Memory

Use separate memory functions instead of treating one text context as "memory":

- working memory: current active state
- episodic memory: events and experiences
- semantic memory: consolidated facts and concepts
- procedural memory: learned action policies
- retrieval/index layer: efficient access to durable memories

The hippocampal/neocortical literature motivates separating event encoding from longer-term consolidation, while the implementation remains an engineering hypothesis. citeturn524900search0turn524900search1

## Affect

Affect is a control signal built from observable/appraised context. Candidate dimensions include valence, arousal, urgency, frustration and confidence. The project must distinguish:
- observed cues
- inferred appraisal
- internal control state
- generated language

This avoids equating an implementation variable with subjective human emotion. citeturn524900search2turn524900search9

## Action and automation

Automation is downstream of cognition:

```
goal → plan → proposed action → permission check → tool execution → observation → verification
```

The learned model does not receive unrestricted operating-system access.

## Learning

The curriculum must be staged and scientifically evaluated:

1. social/language foundations
2. emotion and social reasoning
3. mathematics
4. physics
5. broader natural sciences
6. scientific method and causal reasoning
7. planning and controlled tool use

Coding is excluded from the training objective.

## Development rule

Before adding a module, record:
- scientific motivation
- computational hypothesis
- interfaces/state
- measurable evaluation
- failure modes

One canonical implementation per capability. Legacy code is migrated deliberately rather than duplicated.
