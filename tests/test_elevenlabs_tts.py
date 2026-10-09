"""Tests for the ElevenLabs TARA voice adapter."""
from src.elevenlabs_tts import ElevenLabsConfig, ElevenLabsTTS


class FakeTTS:
    def __init__(self):
        self.calls = []

    class _Speech:
        def convert(self, **kwargs):
            self.parent.calls.append(kwargs)
            return [b"hello ", b"audio"]

    @property
    def text_to_speech(self):
        speech = self._Speech()
        speech.parent = self
        return speech


def test_synthesize_uses_configured_voice_and_model():
    client = FakeTTS()
    config = ElevenLabsConfig(
        api_key="test-key",
        voice_id="voice-123",
        model_id="eleven_v4",
        output_format="mp3_44100_128",
    )
    tts = ElevenLabsTTS(config, client=client)

    assert tts.synthesize("Hello TARA") == b"hello audio"
    assert client.calls == [{
        "text": "Hello TARA",
        "voice_id": "voice-123",
        "model_id": "eleven_v4",
        "output_format": "mp3_44100_128",
    }]


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice")
    monkeypatch.setenv("ELEVENLABS_MODEL", "eleven_v4")
    config = ElevenLabsConfig.from_env()
    assert config is not None
    assert config.voice_id == "voice"
    assert config.model_id == "eleven_v4"


def test_config_from_env_is_disabled_without_credentials(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
    assert ElevenLabsConfig.from_env() is None
