# TARA Chat

Run `python chat_tara.py` from the repository root after a training checkpoint has been created.

The chat client loads the model and Rust-backed BPE tokenizer from the checkpoint itself. It does not call an LLM API.

Useful controls:

- `/clear` resets the conversation context.
- `/exit` quits.

The current 7.3M-parameter model is an early Baby checkpoint, so early outputs can be incoherent until substantially more training is completed.
