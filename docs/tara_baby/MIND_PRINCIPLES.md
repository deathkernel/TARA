# TARA Baby Mind Principles

TARA Baby is a computational cognitive-system research project. Its design target is **human-inspired cognition grounded in science and biology**, not a claim that current engineering can reproduce a perfect human mind.

## Scientific design rule

Every canonical component must answer three questions:

1. **Biological/scientific basis:** what phenomenon or mechanism motivates it?
2. **Computational hypothesis:** what measurable computation are we implementing?
3. **Validation:** what experiment can falsify or support the implementation?

Random "AI features" are not part of the canonical mind architecture.

## Foundations

### Neuroscience and biology
Use neuroscience and biological organization as design constraints and hypotheses for:
- perception and hierarchical processing
- attention and limited working memory
- hippocampal/episodic memory interactions
- prefrontal goal-directed control
- affective appraisal and action selection
- learning, consolidation and forgetting

These mappings are functional analogies, not one-to-one claims about brain areas or consciousness. Research shows that memory involves hippocampal-neocortical interactions and that working memory is limited; the precise computational implementation remains an active research area. citeturn524900search0turn524900search10turn524900search11

### Mathematics
Mathematics is TARA's formal language for:
- probability and uncertainty
- vectors, matrices and optimization
- dynamical state updates
- geometry and spatial relationships
- quantitative prediction
- exact checking of numerical conclusions

### Physics
Physics is used as a source of general principles for:
- state and change over time
- dynamical systems
- constraints and conservation-style checks where applicable
- prediction from observations
- causal/mechanistic models

Physics-inspired ideas are implemented only when they have a clear computational interpretation; a metaphor is not accepted as evidence.

### Cognitive and affective science
For emotion, TARA models appraisal, context, action tendency and regulation signals rather than pretending that a numeric variable is a human feeling. Contemporary research supports links between appraisal and emotion while also showing that emotion does not reduce cleanly to one dedicated brain region. citeturn524900search2turn524900search3turn524900search5

Predictive processing is treated as a testable computational hypothesis rather than a settled description of the whole brain. Research supports predictive processes in neural systems, but the theory remains an area of active investigation. citeturn524900search4turn524900search6

## Canonical cognitive loop

**Observe -> Attend -> Maintain working state -> Update world model -> Infer -> Predict -> Plan -> Act -> Observe consequence -> Compare prediction with outcome -> Correct -> Consolidate memory**

The loop is allowed to revise beliefs and plans when new evidence arrives.

## Non-goals

- No random feature accumulation.
- No coding capability as a training objective.
- No "human brain" claim from benchmark scores alone.
- No treating model size or loss as proof of human-like cognition.
- No biological claim without a clearly stated scientific source or hypothesis.
