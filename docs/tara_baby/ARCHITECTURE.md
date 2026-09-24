# TARA Baby Architecture

## Goal

Build a small but principled neural system that can converse naturally, respond appropriately to conversational affect, reason over mathematics and science, maintain short- and long-term context, plan multi-step tasks, use explicit PC/browser tools, observe outcomes and verify actions.

## Mind loop

User -> perception -> attention -> working memory -> reasoning/planning -> action -> observation -> reflection -> memory

The neural language model is the learned language/reasoning substrate, not the whole mind.

## 1. Neural core

- byte-level/subword tokenizer
- decoder-only Transformer
- RoPE
- RMSNorm
- causal self-attention
- SwiGLU
- tied language head
- AdamW training
- checkpointed experiments

## 2. Cognitive systems

Attention: select the parts of the current context that matter.

Working memory: hold the current conversation, task state, intermediate results and active goals.

Long-term memory: store and retrieve durable facts, preferences and experiences outside the model weights.

Affect/appraisal: represent conversational state such as mood cues, urgency and frustration as behavioral control signals. This is not a claim that TARA experiences human emotion.

Reasoning: combine neural representations with exact numerical, symbolic and causal procedures where appropriate.

Executive control: break requests into goals, subgoals and tool actions.

World model: track entities, state, relationships, time and expected consequences.

Reflection: compare intended actions with observed results and trigger correction.

## 3. Action layer

The neural core must not directly execute arbitrary commands.

A tool policy layer exposes explicit actions such as:
- open_app
- open_url
- read_file
- write_file
- move_file
- type_text
- mouse_click
- press_key
- take_screenshot

The policy layer validates arguments, applies permissions and can stop execution.

## 4. Learning loop

dataset -> audit -> tokenizer -> pretraining/fine-tuning -> validation -> checkpoint -> capability evaluation

Training loss alone is not treated as a measure of a human-like mind.

## 5. Development rule

Prefer one canonical implementation per capability. Before adding a new module, check whether an existing implementation can be migrated, tested and reused.
