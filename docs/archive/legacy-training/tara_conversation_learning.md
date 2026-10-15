> **Historical archive:** preserved for project provenance and phase tracking. These instructions describe a retired/research workflow and are not the current Ollama chat setup. Do not execute commands here as current setup guidance without independently reviewing them.

# TARA Conversation Learning

TARA's interactive CLI supports persistent, user-approved long-term memory. It uses the existing `LongTermMemory` retrieval system and gives relevant saved memories to the configured language model as part of TARA's reasoning context.

## Start

```bat
python -m src
```

By default, memory is stored outside the repository at `~/.tara/memory.json` (on Windows, usually `%USERPROFILE%\.tara\memory.json`). This keeps personal memory out of Git commits. Set `TARA_MEMORY_PATH` to choose a different location.

## Commands

- `/learn <key>=<fact>` — save an approved fact, preference, correction, or project detail.
- `/remember <key>=<fact>` — alias for `/learn`.
- `/memory` — list saved memories.
- `/forget <key>` — delete a memory and persist the change.
- `/help` — show commands.
- `/exit` — quit.

Example:

```text
You: /learn preferred_language=Hinglish
TARA: Learned and saved 'preferred_language'. I'll use it when relevant.
You: /memory
```

When TARA handles later prompts, the existing memory retrieval code adds relevant saved facts to the context sent to Llama through Ollama.

## Learning boundaries

- Saving a memory is explicit; ordinary conversations are not silently written into long-term memory.
- TARA does not inspect Llama's private/raw pretraining dataset. It can use Llama's generated answers as knowledge, but users should verify important claims before saving them.
- This feature changes TARA's external memory, not Llama's model weights. Fine-tuning is a separate, explicit, evaluation-gated workflow.
- Memory is local to the configured path. Treat it as personal data and do not share the file unintentionally.
