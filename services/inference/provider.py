"""Bounded streaming adapter for the configured Groq inference provider.

The upstream hostname is fixed in code (not user-controlled). Usage counters are
process-local; deployments with multiple workers should move budgets to a shared
store before treating them as an account-wide hard limit.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date
import json
import os
from typing import AsyncIterator

import httpx

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"


class ProviderUnavailable(RuntimeError):
    """Provider configuration or upstream service is unavailable."""


@dataclass
class DailyUsage:
    day: date
    requests: int = 0
    tokens: int = 0


class GroqStreamingProvider:
    """OpenAI-compatible Groq SSE adapter with bounded input/output and budgets."""

    def __init__(self) -> None:
        self._usage = DailyUsage(day=date.today())
        self._lock = asyncio.Lock()

    @property
    def configured(self) -> bool:
        return bool(
            os.getenv("GROQ_API_KEY", "").strip()
            and os.getenv("TARA_INFERENCE_API_TOKEN", "").strip()
        )

    def _settings(self) -> tuple[str, str, int, int | None, int | None]:
        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            raise ProviderUnavailable("Inference provider is not configured")
        model = os.getenv("TARA_GROQ_MODEL", "openai/gpt-oss-20b").strip()
        if not model or len(model) > 120:
            raise ProviderUnavailable("Inference model configuration is invalid")
        try:
            completion_cap = int(os.getenv("TARA_GROQ_MAX_COMPLETION_TOKENS", "1024"))
            request_cap = int(os.getenv("TARA_GROQ_MAX_DAILY_REQUESTS", "20"))
            token_cap = int(os.getenv("TARA_GROQ_MAX_DAILY_TOKENS", "50000"))
        except ValueError as exc:
            raise ProviderUnavailable("Inference budget configuration is invalid") from exc
        if not 1 <= completion_cap <= 4096 or request_cap < 1 or token_cap < 1:
            raise ProviderUnavailable("Inference budget configuration is outside allowed bounds")
        return key, model, completion_cap, request_cap, token_cap

    async def _reserve_request(
        self, request_cap: int, token_cap: int, prompt_estimate: int, completion_cap: int
    ) -> tuple[int, int]:
        async with self._lock:
            today = date.today()
            if self._usage.day != today:
                self._usage = DailyUsage(day=today)
            if self._usage.requests >= request_cap:
                raise ProviderUnavailable("Daily inference request budget exhausted")
            remaining_tokens = token_cap - self._usage.tokens
            if remaining_tokens <= prompt_estimate:
                raise ProviderUnavailable("Daily inference token budget exhausted")
            max_tokens = min(completion_cap, remaining_tokens - prompt_estimate)
            reservation = prompt_estimate + max_tokens
            self._usage.requests += 1
            # Reserve an estimated prompt budget plus the maximum completion before
            # opening the upstream request, reducing concurrent budget overshoot.
            self._usage.tokens += reservation
            return max_tokens, reservation

    async def _record_tokens(self, reservation: int, actual_tokens: int) -> None:
        async with self._lock:
            self._usage.tokens = max(0, self._usage.tokens + actual_tokens - reservation)

    async def stream(self, message: str) -> AsyncIterator[dict[str, str]]:
        key, model, completion_cap, request_cap, token_cap = self._settings()
        prompt_estimate = max(1, (len(message) + 2) // 3)
        max_tokens, reservation = await self._reserve_request(
            request_cap, token_cap, prompt_estimate, completion_cap
        )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "max_completion_tokens": max_tokens,
            "temperature": 0.6,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        used_tokens = 0
        saw_usage = False
        emitted_characters = 0
        timeout = httpx.Timeout(45.0, connect=8.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    GROQ_CHAT_COMPLETIONS_URL,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        # Do not forward upstream response bodies or credentials.
                        raise ProviderUnavailable("Inference provider returned an error")
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw:
                            continue
                        if raw == "[DONE]":
                            break
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        usage = chunk.get("usage")
                        if isinstance(usage, dict):
                            total = usage.get("total_tokens")
                            if isinstance(total, int) and total >= 0:
                                used_tokens = total
                                saw_usage = True
                        choices = chunk.get("choices")
                        if not isinstance(choices, list) or not choices:
                            continue
                        first = choices[0]
                        delta = first.get("delta") if isinstance(first, dict) else None
                        text = delta.get("content") if isinstance(delta, dict) else None
                        if isinstance(text, str) and text:
                            emitted_characters += len(text)
                            yield {"type": "token", "token": text}
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ProviderUnavailable("Inference provider could not be reached") from exc
        finally:
            if not saw_usage and emitted_characters:
                used_tokens = prompt_estimate + max(1, (emitted_characters + 3) // 4)
            await self._record_tokens(reservation, used_tokens)


provider = GroqStreamingProvider()
