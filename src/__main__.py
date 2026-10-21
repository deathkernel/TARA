"""Run TARA as an interactive terminal chat with a local Ollama model."""
from __future__ import annotations

import argparse

from .env import load_dotenv
from .brain import TARABrain
from .ollama_provider import OllamaProvider, OllamaProviderError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat with TARA using a local Ollama model."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model name (defaults to OLLAMA_MODEL or llama3.2:3b).",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Ollama server URL (defaults to OLLAMA_HOST or http://localhost:11434).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    provider = OllamaProvider(model=args.model, host=args.host)
    tara = TARABrain(model=None, llm_provider=provider)

    print(f"TARA AI — chat ready (model: {provider.model})")
    print("Commands: /help, /exit")

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTARA closed.")
            return 0

        if not prompt:
            continue
        command = prompt.lower()
        if command in {"/exit", "/quit"}:
            print("TARA closed.")
            return 0
        if command == "/help":
            print("Type a message to chat. Use /exit or /quit to leave.")
            continue

        try:
            response = tara.respond(prompt, max_new_tokens=512, temperature=0.6)
        except OllamaProviderError as exc:
            print(f"\nTARA connection error: {exc}")
            continue
        except Exception as exc:
            print(f"\nTARA error ({type(exc).__name__}): {exc}")
            continue
        print(f"\nTARA: {response.text}")


if __name__ == "__main__":
    raise SystemExit(main())
