# TARA Baby Scientific Curriculum

The curriculum is developmental rather than one giant mixed dataset.

## Stage A — Language and social interaction

Primary sources:
- SODA for broad social dialogue.
- EmpatheticDialogues for emotion-grounded conversation.
- DailyDialog for human-written everyday multi-turn dialogue with explicit emotion and intent labels.
- Blended Skill Talk for combinations of persona, empathy and knowledge.

SODA is useful as a broad social corpus, but its dialogues are synthetic; it should not be treated as a substitute for human-written conversation.

## Stage B — Mathematics and logical structure

- GSM8K for mathematical word-problem reasoning.
- MATH-style competition mathematics after provenance and license review.
- Small generated curricula for arithmetic, algebra, geometry and symbolic transformations with exact validators.

The generated curriculum should include correct and deliberately incorrect examples so TARA learns to distinguish valid reasoning from plausible mistakes.

## Stage C — Physics and causal reasoning

- PHYSICS for multi-domain physics reasoning.
- PhysicsEval or other physics problem sets after provenance and licensing review.
- Public-domain or permissioned educational physics material.

Topic progression:
mechanics -> energy/momentum -> waves -> thermodynamics -> electromagnetism -> optics -> relativity -> quantum foundations.

## Stage D — Natural sciences

Domains:
- chemistry
- biology
- astronomy
- earth science

Use question/answer and explanatory material plus concept-relation graphs and experiment-style examples.

## Stage E — Scientific method

Train examples around:
observation -> hypothesis -> prediction -> experiment -> measurement -> uncertainty/error analysis -> conclusion

TARA should learn that uncertainty and missing evidence are valid outputs.

## Stage F — Tool use and automation

request -> intent -> plan -> tool call -> observation -> verification -> response

Automation data should be structured action traces, not raw command memorization.

## Stage G — Memory and continual learning

Use an external memory system for durable user facts, task history and retrieval. Do not force all personal memory into model weights.

## Mixing policy

Use curriculum stages and small interleaved replay sets to reduce forgetting:
social/conversation -> math -> physics/science -> planning/tool use -> mixed replay/evaluation.

Every stage gets its own checkpoint and evaluation report.
