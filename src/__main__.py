"""Interactive TARA chat using a local language-model provider and persistent memory."""
from __future__ import annotations

import os
from pathlib import Path

from .brain import TARABrain
from .llm_provider_factory import provider_from_env
from .memory import LongTermMemory


class OllamaChatBrain(TARABrain):
    """TARA brain with an explicit error when no trained TARA checkpoint exists."""

    def generate(self, prompt, *, max_new_tokens=32, temperature=1.0, top_k=None, top_p=None):
        raise RuntimeError(
            "Ollama inference failed and no trained TARA checkpoint is configured "
            "as a fallback. Check the Ollama server and model, then retry."
        )


def _save_memory(brain) -> None:
    memory = getattr(brain, "memory", None)
    if memory is None or not hasattr(memory, "save"):
        raise RuntimeError("Persistent memory is not configured for this TARA brain.")
    memory.save()


def chat_loop(brain, *, input_fn=input, output_fn=print) -> None:
    """Run interactive chat, with explicit commands for durable user-approved memory."""
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
            output_fn(
                "Commands: /help, /exit, /learn <key>=<fact>, "
                "/remember <key>=<fact>, /memory, /emotion, /forget <key>"
            )
            continue
        if command == "/emotion":
            label = getattr(brain, "last_emotion", "neutral")
            output_fn(
                f"TARA: Last detected emotional cue: {label}. "
                "This is a lightweight text estimate, not a diagnosis; I do not literally feel emotions."
            )
            continue
        if command == "/memory":
            memory = getattr(brain, "memory", None)
            if memory is None or not hasattr(memory, "keys"):
                output_fn("TARA: Memory is not configured.")
                continue
            keys = memory.keys()
            if not keys:
                output_fn("TARA: My long-term memory is empty.")
                continue
            output_fn("TARA: Saved memories:")
            for key in keys:
                output_fn(f"- {key}: {memory.recall(key)}")
            continue
        if command.startswith("/forget "):
            key = prompt[len("/forget "):].strip()
            memory = getattr(brain, "memory", None)
            if not key or memory is None or not hasattr(memory, "forget"):
                output_fn("TARA: Use /forget <key> with a configured memory.")
                continue
            removed = memory.forget(key)
            if removed is None:
                output_fn(f"TARA: I couldn't find a memory with key '{key}'.")
                continue
            try:
                _save_memory(brain)
            except (OSError, RuntimeError, ValueError) as exc:
                output_fn(f"TARA: Memory was removed for this session but could not be saved: {exc}")
                continue
            output_fn(f"TARA: Forgotten '{key}'.")
            continue
        if command.startswith("/learn ") or command.startswith("/remember "):
            prefix = "/learn " if command.startswith("/learn ") else "/remember "
            entry = prompt[len(prefix):].strip()
            key, separator, value = entry.partition("=")
            key, value = key.strip(), value.strip()
            memory = getattr(brain, "memory", None)
            if not separator or not key or not value or memory is None or not hasattr(memory, "remember"):
                output_fn("TARA: Use /learn <key>=<fact>, for example /learn preferred_language=Hinglish.")
                continue
            try:
                memory.remember(key, value, importance=1.0)
                _save_memory(brain)
            except (OSError, RuntimeError, ValueError) as exc:
                output_fn(f"TARA: I couldn't save that memory: {exc}")
                continue
            output_fn(f"TARA: Learned and saved '{key}'. I'll use it when relevant.")
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
            output_fn(
                "TARA: I couldn't generate a response. Check that Ollama is "
                "running and the configured model is installed. "
                f"Details: {exc}"
            )
            continue
        output_fn(f"TARA: {response.text}")


def main() -> int:
    # For this chat entry point, local Ollama is the sensible default. An
    # explicit TARA_LLM_BACKEND setting still takes precedence.
    os.environ.setdefault("TARA_LLM_BACKEND", "ollama")
    memory_path = Path(
        os.getenv("TARA_MEMORY_PATH", str(Path.home() / ".tara" / "memory.json"))
    ).expanduser()
    try:
        provider = provider_from_env()
        if provider is None:
            print("TARA chat needs a language provider. Set TARA_LLM_BACKEND=ollama.")
            return 2
        memory = LongTermMemory(path=memory_path)
        brain = OllamaChatBrain(model=None, tokenizer=None, llm_provider=provider, memory=memory)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"TARA could not start: {exc}")
        return 2

    print(f"TARA long-term memory: {memory_path}")
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
