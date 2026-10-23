"""Interactive conversation-first TARA CLI."""
from __future__ import annotations

import os

from .brain import TARABrain
from .llm_provider_factory import provider_from_env
from .memory import LongTermMemory


def _display_text(value, *, limit=240) -> str:
    text = "".join(char for char in str(value or "") if char.isprintable())
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _run_memory_command(command_text: str, brain) -> str | None:
    """Handle explicit persistent-memory commands without involving the LLM."""
    parts = command_text.strip().split(maxsplit=1)
    if not parts:
        return None
    verb = parts[0].lower()
    if verb not in {"/memory", "/remember", "/learn", "/forget"}:
        return None
    argument = parts[1].strip() if len(parts) > 1 else ""
    memory = getattr(brain, "memory", None)
    if memory is None:
        return "Memory is not configured for this chat session."

    try:
        if verb == "/memory":
            if argument:
                results = memory.retrieve(argument, limit=5)
                if not results:
                    return "No matching saved memories."
                lines = ["Matching saved memories:"]
                for item in results:
                    if isinstance(item, dict):
                        score = item.get("score")
                        suffix = f" (score: {score:.2f})" if isinstance(score, (int, float)) else ""
                        lines.append(f"- {_display_text(item.get('key'), limit=120)}: {_display_text(item.get('value'), limit=300)}{suffix}")
                return "\n".join(lines)
            keys = memory.keys()
            if not keys:
                return "No saved memories yet. Use /remember key=value to save one."
            lines = ["Saved memories:"]
            for key in keys[:20]:
                lines.append(f"- {_display_text(key, limit=120)}: {_display_text(memory.recall(key), limit=300)}")
            if len(keys) > 20:
                lines.append(f"... and {len(keys) - 20} more.")
            return "\n".join(lines)

        if verb in {"/remember", "/learn"}:
            if "=" not in argument:
                return "Usage: /remember key=value (or /learn key=value)"
            key, value = (part.strip() for part in argument.split("=", 1))
            if not key or not value:
                return "Both memory key and value are required."
            if len(key) > 120 or len(value) > 2000:
                return "Memory key must be at most 120 characters and value at most 2,000 characters."
            memory.remember_and_save(key, value)
            return f"Saved memory: {_display_text(key, limit=120)}"

        if not argument:
            return "Usage: /forget key"
        if len(argument) > 120:
            return "Memory key must be at most 120 characters."
        if argument not in memory.keys():
            return f"No saved memory found for: {_display_text(argument, limit=120)}"
        memory.forget_and_save(argument)
        return f"Forgot memory: {_display_text(argument, limit=120)}"
    except (OSError, TypeError, ValueError) as exc:
        return f"Memory operation failed: {_display_text(exc, limit=240)}"


def chat_loop(brain, *, input_fn=input, output_fn=print) -> None:
    """Run an interactive chat with bounded in-session context."""
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
            output_fn("Commands: /help, /exit, /clear, /memory [query], /remember key=value, /forget key")
            continue
        if command == "/clear":
            clear = getattr(brain, "clear_context", None)
            if callable(clear):
                clear()
                output_fn("TARA: Active conversation context cleared. Saved memories are preserved.")
            else:
                output_fn("TARA: This chat session does not support context clearing.")
            continue
        if command in {"/memory", "/learn", "/remember", "/forget"} or command.startswith(("/memory ", "/learn ", "/remember ", "/forget ")):
            result = _run_memory_command(prompt, brain)
            if result is not None:
                output_fn(f"TARA: {result}")
                continue
        try:
            provider = getattr(brain, "llm_provider", None)
            default_tokens = getattr(provider, "default_max_tokens", 128)
            max_tokens = _positive_int_env("TARA_MAX_RESPONSE_TOKENS", default_tokens)
            response = brain.respond(prompt, max_new_tokens=max_tokens, temperature=0.6)
        except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
            output_fn(f"TARA: I couldn't generate a response. Check the configured language provider. Details: {_display_text(exc)}")
            continue
        output_fn(f"TARA: {response.text}")


def main() -> int:
    os.environ.setdefault("TARA_LLM_BACKEND", "ollama")
    try:
        provider = provider_from_env()
        if provider is None:
            print("TARA chat needs a language provider. Set TARA_LLM_BACKEND=ollama.")
            return 2
        memory_path = os.getenv("TARA_MEMORY_PATH", "data/tara_memory.json").strip()
        if not memory_path:
            raise ValueError("TARA_MEMORY_PATH must not be empty")
        max_history_turns = _positive_int_env("TARA_MAX_HISTORY_TURNS", 12)
        language = os.getenv("TARA_LANGUAGE", "auto")
        personality = os.getenv("TARA_PERSONALITY", "balanced")
        brain = TARABrain(
            llm_provider=provider, memory=LongTermMemory(memory_path),
            max_history_turns=max_history_turns, language=language, personality=personality,
        )
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
    print(f"TARA style: language={language}, personality={personality}, history_turns={max_history_turns}")
    chat_loop(brain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
