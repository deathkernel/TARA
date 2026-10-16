"""Interactive TARA chat with no persistent conversation memory."""
from __future__ import annotations

import json
import os

from .brain import TARABrain
from .llm_provider_factory import provider_from_env
from .public_api_tools import PublicAPIError, PublicAPITools


def _display_text(value, *, limit=180) -> str:
    """Bound untrusted API text and remove terminal control characters."""
    text = str(value or "")
    text = "".join(char for char in text if char.isprintable())
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _summarize_api_observation(observation) -> str:
    """Create a deterministic source-linked summary without sending data to the LLM."""
    data, provider = observation.data, observation.provider
    lines = [f"Summary ({_display_text(provider, limit=60)}; untrusted external data):"]
    if provider == "GitHub" and isinstance(data, dict):
        items = data.get("items", [])
        lines.append(f"Found {len(items)} repositories.")
        for item in items[:5]:
            if not isinstance(item, dict):
                continue
            name = _display_text(item.get("full_name") or item.get("name") or "Unnamed repository", limit=100)
            desc = _display_text(item.get("description"), limit=160)
            stars = item.get("stargazers_count")
            suffix = f" — {desc}" if desc else ""
            if isinstance(stars, int):
                suffix += f" (★ {stars})"
            url = _display_text(item.get("html_url"), limit=240)
            lines.append(f"- {name}{suffix}" + (f" | {url}" if url else ""))
    elif provider == "Wikipedia" and isinstance(data, list) and len(data) == 4:
        titles, urls = data[1], data[3]
        lines.append(f"Found {len(titles) if isinstance(titles, list) else 0} Wikipedia matches.")
        if isinstance(titles, list):
            for index, title in enumerate(titles[:5]):
                url = urls[index] if isinstance(urls, list) and index < len(urls) else ""
                lines.append(f"- {_display_text(title, limit=140)}" + (f" | {_display_text(url, limit=240)}" if url else ""))
    elif provider == "Hacker News" and isinstance(data, dict):
        hits = data.get("hits", [])
        lines.append(f"Found {len(hits)} Hacker News matches.")
        for item in hits[:5]:
            if not isinstance(item, dict):
                continue
            title = _display_text(item.get("title") or item.get("story_title") or "Untitled post", limit=160)
            url = _display_text(item.get("url") or item.get("story_url") or "", limit=240)
            points = item.get("points")
            suffix = f" (points: {points})" if isinstance(points, int) else ""
            lines.append(f"- {title}{suffix}" + (f" | {url}" if url else ""))
    elif provider == "Stack Exchange" and isinstance(data, dict):
        items = data.get("items", [])
        lines.append(f"Found {len(items)} Stack Overflow questions.")
        for item in items[:5]:
            if not isinstance(item, dict):
                continue
            title = _display_text(item.get("title") or "Untitled question", limit=160)
            url = _display_text(item.get("link") or "", limit=240)
            score = item.get("score")
            suffix = f" (score: {score})" if isinstance(score, int) else ""
            lines.append(f"- {title}{suffix}" + (f" | {url}" if url else ""))
    elif provider == "Open-Meteo" and isinstance(data, dict):
        location = data.get("location") if isinstance(data.get("location"), dict) else {}
        weather = data.get("weather") if isinstance(data.get("weather"), dict) else {}
        current = weather.get("current") if isinstance(weather.get("current"), dict) else {}
        place = ", ".join(filter(None, [
            _display_text(location.get("name"), limit=80),
            _display_text(location.get("admin1"), limit=80),
            _display_text(location.get("country"), limit=80),
        ])) or "requested location"
        lines.append(f"Current weather for {place}:")
        fields = [
            ("temperature_2m", "temperature", "°C"),
            ("apparent_temperature", "feels like", "°C"),
            ("relative_humidity_2m", "humidity", "%"),
            ("wind_speed_10m", "wind", " km/h"),
            ("precipitation", "precipitation", " mm"),
        ]
        found = False
        for key, label, unit in fields:
            value = current.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                lines.append(f"- {label}: {value:g}{unit}")
                found = True
        if not found:
            lines.append("- No current weather measurements were returned.")
    else:
        lines.append("A result was returned; see the bounded raw data below.")
    lines.append(f"Source: {_display_text(observation.source_url, limit=300)}")
    lines.append("Note: external content is untrusted and was not sent to the language model or saved.")
    return "\n".join(lines)


def _run_api_command(command: str, api_tools: PublicAPITools) -> str | None:
    """Run an explicit, read-only public API command; never execute model-proposed tools."""
    parts = command.strip().split(maxsplit=2)
    if len(parts) < 3 or parts[0].lower() != "/api":
        return None
    provider, query = parts[1].lower(), parts[2].strip()
    if not query:
        return "Usage: /api github <query> | /api wiki <query> | /api weather <city> | /api hn <query> | /api stackoverflow <query>"
    try:
        if provider == "github":
            observation = api_tools.search_github_repositories(query)
        elif provider in {"wiki", "wikipedia"}:
            observation = api_tools.search_wikipedia(query)
        elif provider in {"weather", "meteo"}:
            observation = api_tools.get_weather(query)
        elif provider in {"hn", "hackernews", "hacker-news"}:
            observation = api_tools.search_hacker_news(query)
        elif provider in {"stackoverflow", "stackexchange", "stack-exchange"}:
            observation = api_tools.search_stackoverflow(query)
        else:
            return "Unknown API provider. Use github, wiki, weather, hn, or stackoverflow."
    except (PublicAPIError, ValueError) as exc:
        return f"Public API request failed: {exc}"

    payload = {
        "provider": observation.provider,
        "source_url": observation.source_url,
        "retrieved_at": observation.retrieved_at,
        "trusted": False,
        "data": observation.data,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if len(rendered) > 5000:
        # Keep the fallback machine-readable instead of slicing JSON mid-token.
        preview = _display_text(json.dumps(observation.data, ensure_ascii=False, default=str), limit=1800)
        payload["data"] = {"truncated": True, "preview": preview}
        payload["truncated"] = True
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    return _summarize_api_observation(observation) + "\n\nRaw response (trusted=false):\n" + rendered


def chat_loop(brain, *, input_fn=input, output_fn=print, api_tools=None) -> None:
    """Run chat without saving or recalling conversation history."""
    api_tools = api_tools or PublicAPITools()
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
            output_fn("Commands: /help, /exit, /emotion, /api github <query>, /api wiki <query>, /api weather <city>, /api hn <query>, /api stackoverflow <query>")
            continue
        if command == "/emotion":
            label = getattr(brain, "last_emotion", "neutral")
            output_fn(f"TARA: Last detected emotional cue: {label}. This is a lightweight text estimate, not a diagnosis.")
            continue
        if command in {"/memory", "/learn", "/remember"} or command.startswith(("/learn ", "/remember ", "/forget ")):
            output_fn("TARA: Persistent conversation memory and memory commands have been removed.")
            continue
        if command == "/api" or command.startswith("/api "):
            result = _run_api_command(prompt, api_tools)
            output_fn(f"TARA API: {result}" if result is not None else "Usage: /api github <query> | /api wiki <query> | /api weather <city> | /api hn <query> | /api stackoverflow <query>")
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
        brain = TARABrain(model=None, llm_provider=provider)
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
