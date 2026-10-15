"""Interactive TARA chat with no persistent conversation memory."""
from __future__ import annotations

import os

from .brain import TARABrain
from .llm_provider_factory import provider_from_env


class OllamaChatBrain(TARABrain):
    """TARA brain that relies on its configured language provider only."""

    def generate(self, prompt, *, max_new_tokens=32, temperature=1.0, top_k=None, top_p=None):
        raise RuntimeError("No local TARA checkpoint fallback is available; use the configured language provider.")


def chat_loop(brain, *, input_fn=input, output_fn=print) -> None:
    """Run chat without saving or recalling conversation history."""
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
            output_fn("Commands: /help, /exit, /emotion")
            continue
        if command == "/emotion":
            label = getattr(brain, "last_emotion", "neutral")
            output_fn(f"TARA: Last detected emotional cue: {label}. This is a lightweight text estimate, not a diagnosis.")
            continue
        if command in {"/memory", "/learn", "/remember"} or command.startswith(("/learn ", "/remember ", "/forget ")):
            output_fn("TARA: Persistent conversation memory and memory commands have been removed.")
            continue
        try:
            provider = getattr(brain, "llm_provider", None)
            default_tokens = getattr(provider, "default_max_tokens", 128)
            raw_tokens = os.getenv("TARA_MAX_RESPONSE_TOKENS", "").strip()
            if raw_tokens:
                try:
                    max_tokens = int(raw_tokens)
                except ValueError as exc:
                    raise ValueError("TARA_MAX_RESPONSE_TOKENS must be a positive integer") from exc
                if max_tokens <= 0:
                    raise ValueError("TARA_MAX_RESPONSE_TOKENS must be a positive integer")
            else:
                max_tokens = default_tokens
            response = brain.respond(prompt, max_new_tokens=max_tokens, temperature=0.6)
        except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
            output_fn(f"TARA: I couldn't generate a response. Check the configured language provider. Details: {exc}")
            continue
        output_fn(f"TARA: {response.text}")


def main() -> int:
    os.environ.setdefault("TARA_LLM_BACKEND", "ollama")
    try:
        provider = provider_from_env()
        if provider is None:
            print("TARA chat needs a language provider. Set TARA_LLM_BACKEND=ollama.")
            return 2
        brain = OllamaChatBrain(model=None, tokenizer=None, llm_provider=provider)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"TARA could not start: {exc}")
        return 2
    print(
        "TARA resource profile: "
        f"{getattr(provider, 'power_mode', 'custom')} "
        f"(context={getattr(provider, 'num_ctx', 'default')}, "
        f"CPU threads={getattr(provider, 'num_thread', 'auto')}, "
        f"batch={getattr(provider, 'num_batch', 'default')}, "
        f"max response tokens={getattr(provider, 'default_max_tokens', 128)})"
    )
    chat_loop(brain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
