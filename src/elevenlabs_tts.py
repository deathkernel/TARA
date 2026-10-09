"""Optional ElevenLabs text-to-speech provider for TARA.

TARA owns cognition and response generation; this module only converts
already-generated response text into speech through ElevenLabs.
The API key and voice ID are read from the environment and never stored
in the repository.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ElevenLabsConfig:
    api_key: str
    voice_id: str
    model_id: str = "eleven_v4"
    output_format: str = "mp3_44100_128"

    @classmethod
    def from_env(cls) -> "ElevenLabsConfig | None":
        api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "4cLxTzxGs6YiIomdhhqO").strip()
        if not api_key or not voice_id:
            return None
        return cls(
            api_key=api_key,
            voice_id=voice_id,
            model_id=os.getenv("ELEVENLABS_MODEL", "eleven_v4").strip() or "eleven_v4",
            output_format=os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128").strip()
            or "mp3_44100_128",
        )


class ElevenLabsTTS:
    """Small, injectable ElevenLabs TTS adapter for TARA."""

    def __init__(self, config: ElevenLabsConfig, *, client: Any | None = None) -> None:
        if not config.api_key:
            raise ValueError("ELEVENLABS_API_KEY is required")
        if not config.voice_id:
            raise ValueError("ELEVENLABS_VOICE_ID is required")
        self.config = config
        if client is None:
            try:
                from elevenlabs.client import ElevenLabs
            except ImportError as exc:
                raise RuntimeError(
                    "ElevenLabs support requires the 'elevenlabs' package"
                ) from exc
            client = ElevenLabs(api_key=config.api_key)
        self.client = client

    @classmethod
    def from_env(cls) -> "ElevenLabsTTS | None":
        config = ElevenLabsConfig.from_env()
        return None if config is None else cls(config)

    def synthesize(self, text: str) -> bytes:
        """Generate audio bytes for text using the configured ElevenLabs voice."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        audio = self.client.text_to_speech.convert(
            text=text,
            voice_id=self.config.voice_id,
            model_id=self.config.model_id,
            output_format=self.config.output_format,
        )

        if isinstance(audio, bytes):
            return audio
        return b"".join(audio)

    def speak(self, text: str) -> bytes:
        """Generate and play speech, returning the generated audio bytes."""
        audio = self.synthesize(text)
        try:
            from elevenlabs.play import play
        except ImportError as exc:
            raise RuntimeError(
                "ElevenLabs playback requires the 'elevenlabs' package"
            ) from exc
        play(audio)
        return audio


__all__ = ["ElevenLabsConfig", "ElevenLabsTTS"]
