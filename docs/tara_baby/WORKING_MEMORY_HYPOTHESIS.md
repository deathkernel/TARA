# Working Memory Hypothesis

## Scientific motivation

Working memory is widely modeled as a limited-capacity system that maintains and manipulates information for current behavior, with strong interactions with attention and long-term knowledge. Different theories disagree on the exact mechanism and capacity; interference and resource limits remain active research topics. citeturn325979search1turn325979search2turn325979search6

A 2026 review continues to develop multicomponent working-memory models, so TARA treats its implementation as a testable engineering hypothesis rather than a settled biological recipe. citeturn325979search5

## Computational hypothesis

TARA uses a bounded store of active representations. Each item has explicit priority plus age and reuse signals.

retention = wp*priority + wr*recency + wu*reuse

The weights sum to one. When capacity is full, the lowest-retention item is replaced.

Default capacity is 7 only as an engineering starting point, not as a claim that human working memory universally contains exactly seven items. Scientific literature contains competing capacity estimates and mechanisms. citeturn325979search2turn325979search9

## Biological grounding

Prefrontal networks are strongly implicated in maintaining task-relevant information and goals, while working memory interacts with attention and long-term memory. These findings motivate TARA's separation of active state from durable memory, but no software slot is treated as a literal neuron or brain region. citeturn325979search4turn325979search8turn325979search0

## Validation

Initial tests verify bounded capacity, priority-sensitive retention, attentional refreshing, deterministic ordering, and invalid-parameter rejection.

Future behavioral tests should measure distractor resistance, interference, goal switching, multi-step retention, and transfer from episodic and semantic memory.
