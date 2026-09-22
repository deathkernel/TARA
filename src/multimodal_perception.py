"""Multimodal perception and fusion for TARA.

The implementation is dependency-light and honest about its boundary:
raw modalities are normalized into structured observations and deterministic
feature vectors. Attention-style fusion combines them without pretending that
a learned vision/audio encoder exists when one has not been trained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import exp, sqrt
from pathlib import Path
import struct
import wave
from typing import Any, Iterable, Mapping, Sequence

from .perception import summarize_screen_state


MODALITIES = ("text", "document", "image", "audio", "screen")


@dataclass(frozen=True)
class MultimodalObservation:
    observation_id: str
    modality: str
    content: str
    features: tuple[float, ...]
    confidence: float
    metadata: tuple[tuple[str, str], ...] = ()
    source: str = "multimodal"


@dataclass(frozen=True)
class FusedRepresentation:
    vector: tuple[float, ...]
    modality_weights: tuple[tuple[str, float], ...]
    dominant_modality: str
    confidence: float
    fingerprint: str


@dataclass(frozen=True)
class MultimodalContext:
    observations: tuple[MultimodalObservation, ...]
    fused: FusedRepresentation
    summary: str


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    format: str
    byte_count: int


@dataclass(frozen=True)
class AudioInfo:
    channels: int
    sample_rate: int
    frames: int
    duration_seconds: float
    sample_width: int


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _fingerprint(parts: Iterable[str]) -> str:
    return sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _unit_vector(values: Sequence[float]) -> tuple[float, ...]:
    norm = sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        return tuple(0.0 for _ in values)
    return tuple(value / norm for value in values)


class HashedFeatureEncoder:
    """Deterministic feature hashing for text-like and binary observations."""

    def __init__(self, dimensions: int = 32):
        if dimensions < 8:
            raise ValueError("dimensions must be >= 8")
        self.dimensions = dimensions

    def _hash_slot(self, token: str) -> tuple[int, int]:
        digest = sha256(token.encode("utf-8")).digest()
        slot = int.from_bytes(digest[:4], "big") % self.dimensions
        sign = 1 if digest[4] & 1 else -1
        return slot, sign

    def encode_text(self, text: str) -> tuple[float, ...]:
        tokens = [token for token in text.lower().split() if token]
        features = [0.0] * self.dimensions
        if not tokens:
            return tuple(features)
        for token in tokens:
            slot, sign = self._hash_slot(token)
            features[slot] += sign
            for ngram in (token[i:i + 3] for i in range(max(0, len(token) - 2))):
                slot, sign = self._hash_slot("char:" + ngram)
                features[slot] += 0.25 * sign
        return _unit_vector(features)

    def encode_bytes(self, data: bytes) -> tuple[float, ...]:
        features = [0.0] * self.dimensions
        if not data:
            return tuple(features)
        for offset in range(0, len(data), max(1, len(data) // 256)):
            chunk = data[offset:offset + 32]
            slot, sign = self._hash_slot(sha256(chunk).hexdigest())
            features[slot] += sign
        return _unit_vector(features)

    def encode_numeric(self, values: Sequence[float]) -> tuple[float, ...]:
        vector = [0.0] * self.dimensions
        if not values:
            return tuple(vector)
        for index, value in enumerate(values):
            vector[index % self.dimensions] += float(value)
        return _unit_vector(vector)


class CrossModalAttentionFusion:
    """Confidence-gated late fusion inspired by attention weighting.

    This is not a trained attention layer: weights are deterministic from
    modality confidence and feature energy until learned encoders are plugged in.
    """

    def __init__(self, temperature: float = 0.75):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def fuse(self, observations: Sequence[MultimodalObservation]) -> FusedRepresentation:
        if not observations:
            empty = tuple()
            return FusedRepresentation(empty, (), "", 0.0, _fingerprint(()))
        logits = []
        for item in observations:
            energy = sqrt(sum(value * value for value in item.features))
            logits.append((_clip(item.confidence) * max(energy, 1e-6)) / self.temperature)
        maximum = max(logits)
        exp_values = [exp(value - maximum) for value in logits]
        total = sum(exp_values)
        weights = [value / total for value in exp_values]
        dimensions = max(len(item.features) for item in observations)
        fused = [0.0] * dimensions
        for weight, item in zip(weights, observations):
            for index, value in enumerate(item.features):
                fused[index] += weight * value
        fused_vector = _unit_vector(fused)
        modality_weights = tuple((item.modality, round(weight, 8)) for item, weight in zip(observations, weights))
        dominant_index = max(range(len(observations)), key=lambda i: weights[i])
        confidence = _clip(sum(weight * item.confidence for weight, item in zip(weights, observations)))
        fingerprint = _fingerprint(
            [item.observation_id + ":" + str(round(weight, 8)) for item, weight in zip(observations, weights)]
        )
        return FusedRepresentation(
            fused_vector,
            modality_weights,
            observations[dominant_index].modality,
            confidence,
            fingerprint,
        )


class MultimodalPerception:
    def __init__(self, *, dimensions: int = 32, fusion_temperature: float = 0.75):
        self.encoder = HashedFeatureEncoder(dimensions)
        self.fusion = CrossModalAttentionFusion(fusion_temperature)

    def _observation(
        self,
        modality: str,
        content: str,
        features: Sequence[float],
        *,
        confidence: float,
        metadata: Mapping[str, Any] | None = None,
        source: str,
    ) -> MultimodalObservation:
        if modality not in MODALITIES:
            raise ValueError(f"unsupported modality: {modality}")
        confidence = _clip(confidence)
        normalized_features = tuple(float(value) for value in features)
        meta = tuple((str(key), str(value)) for key, value in sorted((metadata or {}).items()))
        observation_id = _fingerprint(
            [modality, content, str(normalized_features), str(meta), str(round(confidence, 8))]
        )[:24]
        return MultimodalObservation(
            observation_id, modality, content[:4096], normalized_features,
            confidence, meta, source,
        )

    def text(self, text: str, *, confidence: float = 1.0, source: str = "text") -> MultimodalObservation:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self._observation(
            "text", text, self.encoder.encode_text(text),
            confidence=confidence, source=source,
        )

    def document(self, text: str, *, path: str = "", confidence: float = 1.0) -> MultimodalObservation:
        observation = self.text(text, confidence=confidence, source="document")
        metadata = {"path": path, "length": len(text)}
        return self._observation(
            "document", observation.content, observation.features,
            confidence=observation.confidence, metadata=metadata, source="document",
        )

    def image(self, data: bytes, *, source: str = "image", confidence: float = 0.8) -> MultimodalObservation:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("image data must be bytes")
        info = self._image_info(bytes(data))
        features = self.encoder.encode_bytes(bytes(data))
        return self._observation(
            "image",
            f"{info.format} {info.width}x{info.height} image",
            features,
            confidence=confidence,
            metadata=info.__dict__,
            source=source,
        )

    def audio(self, data: bytes, *, source: str = "audio", confidence: float = 0.7) -> MultimodalObservation:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("audio data must be bytes")
        info = self._audio_info(bytes(data))
        features = self.encoder.encode_numeric(
            [info.channels, info.sample_rate / 48000.0, info.duration_seconds, info.sample_width]
        )
        return self._observation(
            "audio",
            f"WAV {info.channels}ch {info.sample_rate}Hz {info.duration_seconds:.3f}s",
            features,
            confidence=confidence,
            metadata=info.__dict__,
            source=source,
        )

    def screen(self, elements: Sequence[Mapping[str, Any]], *, confidence: float = 0.9) -> MultimodalObservation:
        summary = summarize_screen_state(elements)
        return self._observation(
            "screen", summary, self.encoder.encode_text(summary),
            confidence=confidence, metadata={"element_count": len(elements)}, source="screen",
        )

    def perceive(self, observations: Sequence[MultimodalObservation]) -> MultimodalContext:
        if not observations:
            raise ValueError("at least one observation is required")
        fused = self.fusion.fuse(observations)
        summary_parts = [
            f"{item.modality}[{item.confidence:.2f}]: {item.content}" for item in observations
        ]
        summary = "\n".join(summary_parts)
        return MultimodalContext(tuple(observations), fused, summary)

    def _image_info(self, data: bytes) -> ImageInfo:
        if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
            width, height = struct.unpack(">II", data[16:24])
            return ImageInfo(width, height, "png", len(data))
        if data.startswith(b"\xff\xd8"):
            width, height = self._jpeg_dimensions(data)
            return ImageInfo(width, height, "jpeg", len(data))
        raise ValueError("unsupported image format; expected PNG or JPEG")

    def _jpeg_dimensions(self, data: bytes) -> tuple[int, int]:
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            index += 2
            if marker in (0xD8, 0xD9):
                continue
            if index + 2 > len(data):
                break
            length = struct.unpack(">H", data[index:index + 2])[0]
            if 0xC0 <= marker <= 0xC3 and index + 7 <= len(data):
                height, width = struct.unpack(">HH", data[index + 3:index + 7])
                return width, height
            index += max(2, length)
        raise ValueError("JPEG dimensions could not be parsed")

    def _audio_info(self, data: bytes) -> AudioInfo:
        import io
        try:
            with wave.open(io.BytesIO(data), "rb") as wav:
                channels = wav.getnchannels()
                sample_rate = wav.getframerate()
                frames = wav.getnframes()
                sample_width = wav.getsampwidth()
        except wave.Error as exc:
            raise ValueError("unsupported audio format; expected PCM WAV") from exc
        duration = frames / sample_rate if sample_rate else 0.0
        return AudioInfo(channels, sample_rate, frames, duration, sample_width)
