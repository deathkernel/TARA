from src.multimodal_perception import MultimodalObservation, MultimodalPerception


def test_text_screen_and_fusion_are_deterministic():
    engine = MultimodalPerception(dimensions=16)
    text = engine.text("hello world", confidence=1.0)
    screen = engine.screen([{"type": "button", "label": "Login"}], confidence=0.8)
    context = engine.perceive([text, screen])
    assert context.fused.vector
    assert context.fused.dominant_modality in {"text", "screen"}
    assert sum(weight for _, weight in context.fused.modality_weights) == 1.0
    assert engine.perceive([text, screen]).fused.fingerprint == context.fused.fingerprint


def test_image_png_metadata():
    import struct
    data = b"\\x89PNG\\r\\n\\x1a\\n" + b"0" * 8 + struct.pack(">II", 64, 32)
    observation = MultimodalPerception().image(data)
    assert observation.modality == "image"
    assert dict(observation.metadata)["width"] == "64"


def test_audio_wav_metadata():
    import io
    import wave
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\\x00\\x00" * 800)
    observation = MultimodalPerception().audio(out.getvalue())
    assert observation.modality == "audio"
    assert dict(observation.metadata)["sample_rate"] == "8000"


def test_observation_shape():
    obs = MultimodalObservation("id", "text", "hello", (1.0,), 1.0)
    assert obs.observation_id == "id"
