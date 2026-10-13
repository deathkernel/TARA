"""Interactive TARA chat using the configured language-model provider.

Run from the repository root with `python -m src`. The CLI defaults to the
local Ollama provider so the user talks to TARA, not to Ollama's own shell.
"""
from __future__ import annotations

import os

from .brain import TARABrain
from .llm_provider_factory import provider_from_env


def chat_loop(brain, *, input_fn=input, output_fn=print) -> None:
    """Run a small interactive terminal chat loop; injectable for unit tests."""
    output_fn("TARA chat is ready. Type /help for commands, /exit to quit.")
    while True:
        try:
            prompt = input_fn("You: ")
        except (EOFError, KeyboardInterrupt):
            output_fn("\nTARA: Chat closed.")
            return

        prompt = prompt.strip()
        if not prompt:
            continue
        command = prompt.lower()
        if command in {"/exit", "/quit", "/bye"}:
            output_fn("TARA: Bye, bhai!")
            return
        if command == "/help":
            output_fn("Commands: /help, /exit")
            continue

        try:
            response = brain.respond(prompt, max_new_tokens=256, temperature=0.6)
        except (RuntimeError, TimeoutError, OSError) as exc:
            output_fn(
                "TARA: I couldn't reach the configured language model. "
                "Check that Ollama is running and the model is installed. "
                f"Details: {exc}"
            )
            continue
        output_fn(f"TARA: {response.text}")


def main() -> int:
    # For this chat entry point, local Ollama is the sensible default. An
    # explicit TARA_LLM_BACKEND setting still takes precedence.
    os.environ.setdefault("TARA_LLM_BACKEND", "ollama")
    try:
        provider = provider_from_env()
        if provider is None:
            print("TARA chat needs a language provider. Set TARA_LLM_BACKEND=ollama.")
            return 2
        brain = TARABrain(model=None, tokenizer=None, llm_provider=provider)
    except (ValueError, RuntimeError) as exc:
        print(f"TARA could not start: {exc}")
        return 2

    chat_loop(brain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
