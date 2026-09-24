"""Groq-backed LLM provider for TARA.

Groq is an optional inference backend. TARA remains responsible for memory,
planning, verification, and action; this provider only supplies language/reasoning
when TARA chooses to call an external LLM.

The API key is read from GROQ_API_KEY and is never stored in the repository.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import os
from typing import Any

from .env import load_dotenv

load_dotenv()


@dataclass
class GroqUsage:
    requests: int = 0
    tokens: int = 0


class GroqProvider:
    """Small, budget-aware Groq adapter with a TARA-friendly interface."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "openai/gpt-oss-20b",
        max_daily_requests: int | None = None,
        max_daily_tokens: int | None = None,
        max_completion_tokens: int = 1024,
        client: Any = None,
    ):
        if max_daily_requests is not None and max_daily_requests <= 0:
            raise ValueError("max_daily_requests must be positive")
        if max_daily_tokens is not None and max_daily_tokens <= 0:
            raise ValueError("max_daily_tokens must be positive")
        if max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")

        self.model = model
        self.max_daily_requests = max_daily_requests
        self.max_daily_tokens = max_daily_tokens
        self.max_completion_tokens = max_completion_tokens
        self._day = date.today()
        self.usage = GroqUsage()

        if client is not None:
            self.client = client
        else:
            key = api_key or os.getenv("GROQ_API_KEY")
            if not key:
                raise ValueError(
                    "GROQ_API_KEY is required to use GroqProvider"
                )
            try:
                from groq import Groq
            except ImportError as exc:
                raise ImportError(
                    "The 'groq' package is required. Install it with: pip install groq"
                ) from exc
            self.client = Groq(api_key=key)

    def _reset_if_new_day(self) -> None:
        today = date.today()
        if today != self._day:
            self._day = today
            self.usage = GroqUsage()

    def _budget_available(self) -> bool:
        self._reset_if_new_day()
        if (
            self.max_daily_requests is not None
            and self.usage.requests >= self.max_daily_requests
        ):
            return False
        if (
            self.max_daily_tokens is not None
            and self.usage.tokens >= self.max_daily_tokens
        ):
            return False
        return True

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_completion_tokens: int | None = None,
        temperature: float = 0.6,
    ) -> str:
        """Ask Groq for one response, respecting TARA's local daily budget."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not self._budget_available():
            raise RuntimeError("TARA Groq daily budget exhausted")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        requested_max = max_completion_tokens or self.max_completion_tokens
        if requested_max <= 0:
            raise ValueError("max_completion_tokens must be positive")
        remaining_tokens = self.remaining_daily_tokens
        effective_max = min(requested_max, self.max_completion_tokens)
        if remaining_tokens is not None:
            if remaining_tokens <= 0:
                raise RuntimeError("TARA Groq daily token budget exhausted")
            effective_max = min(effective_max, remaining_tokens)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=effective_max,
        )
        text = response.choices[0].message.content or ""

        self.usage.requests += 1
        usage = getattr(response, "usage", None)
        total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
        self.usage.tokens += int(total_tokens or 0)
        return text

    @property
    def remaining_daily_requests(self) -> int | None:
        self._reset_if_new_day()
        if self.max_daily_requests is None:
            return None
        return max(0, self.max_daily_requests - self.usage.requests)

    @property
    def remaining_daily_tokens(self) -> int | None:
        self._reset_if_new_day()
        if self.max_daily_tokens is None:
            return None
        return max(0, self.max_daily_tokens - self.usage.tokens)
