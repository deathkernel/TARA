# TARA Curriculum v1

This is a foundation-first teaching dataset for the small local TARA language model.

The curriculum intentionally avoids story-heavy training. Each record is a compact lesson made from direct question/answer examples, progressing from simple language to arithmetic, facts, cause/effect, instructions, respectful communication, memory/context, and basic reasoning.

## Order

1. Language foundations
2. Numbers and counting
3. Basic arithmetic
4. Objects and properties
5. Colors, shapes, and patterns
6. Nature and science basics
7. Time and everyday knowledge
8. Cause and effect
9. Following instructions
10. Emotions and respectful communication
11. Memory and context
12. Simple reasoning and learning

The records are kept long enough for the current 128-token training context, while each lesson remains internally coherent.

## Training

Use the normal public CLI:

```text
python tara.py train data/curriculum_v1.jsonl
python tara.py chat
```

The training pipeline samples windows **inside individual records** rather than flattening separate lessons into one token stream. This prevents one lesson from teaching the model accidental transitions from the end of one conversation into the beginning of another.

## Design rule

This dataset is a foundation, not the final communication corpus. After the model learns the basic curriculum, later stages can introduce carefully selected natural conversations, richer context, reasoning examples, and eventually broader knowledge.
