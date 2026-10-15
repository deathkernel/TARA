"""Lightweight, local emotional-cue detection for TARA.

This is a transparent keyword heuristic, not a clinical classifier or evidence
that TARA has subjective feelings. It adds no model download or inference pass.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmotionAssessment:
    label: str
    intensity: float
    evidence: tuple[str, ...]
    response_guidance: str


# Keep the first version small and inspectable. Include common English and
# Hinglish/Hindi transliterations; users can extend these vocabularies later.
_CUES = {
    "sadness": {
        "sad", "unhappy", "depressed", "heartbroken", "cry", "crying", "hurt",
        "lonely", "hopeless", "upset", "dukhi", "udaas", "akela", "akeli",
        "rona", "ro raha", "ro rahi", "dil toot", "bura lag", "pareshan",
    },
    "anxiety": {
        "anxious", "anxiety", "scared", "afraid", "worried", "panic", "nervous",
        "stress", "stressed", "tension", "darr", "dar lag", "ghabra", "chinta",
        "fikar", "exam ka dar",
    },
    "anger": {
        "angry", "furious", "mad", "hate", "annoyed", "irritated", "gussa",
        "ghussa", "naraz", "nafrat", "bakwas", "fed up",
    },
    "frustration": {
        "frustrated", "stuck", "failed", "failure", "not working", "doesn't work",
        "samajh nahi", "samajh nhi", "nahi ho raha", "nhi ho raha", "thak gaya",
        "thak gayi", "fed up", "confusing",
    },
    "joy": {
        "happy", "glad", "great", "wonderful", "awesome", "love it", "excited",
        "khush", "maza aa", "bahut acha", "bohot achha", "mast", "yay", "woohoo",
    },
    "gratitude": {
        "thank you", "thanks", "thankful", "grateful", "shukriya", "dhanyavaad",
        "thankyou", "thanks bhai",
    },
    "confusion": {
        "confused", "don't understand", "do not understand", "what does this mean",
        "samajh nahi aaya", "samajh nahi aa raha", "kya matlab", "kaise",
    },
}

_GUIDANCE = {
    "sadness": "Be gentle and validating. Listen first; offer one small practical next step without minimizing their pain.",
    "anxiety": "Be calm and grounding. Break the issue into manageable steps and avoid alarmist language.",
    "anger": "Stay respectful and non-defensive. Acknowledge the frustration without mirroring hostility.",
    "frustration": "Acknowledge the obstacle. Be practical, concise, and give the next actionable step rather than generic reassurance.",
    "joy": "Share the positive tone naturally without exaggerating or pretending to have human feelings.",
    "gratitude": "Respond warmly and briefly; do not make the exchange about yourself.",
    "confusion": "Explain simply, step by step, and ask one focused clarifying question only if needed.",
    "neutral": "Be natural, helpful, and direct. Match the user's language and level of detail.",
}


class EmotionalIntelligence:
    """Fast, deterministic emotion-cue detector with no extra model inference."""

    def assess(self, text: str) -> EmotionAssessment:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        normalized = " ".join(text.lower().split())
        scores: dict[str, list[str]] = {label: [] for label in _CUES}
        for label, cues in _CUES.items():
            for cue in cues:
                if cue in normalized:
                    scores[label].append(cue)

        # Frustration often overlaps with confusion; prioritize the most useful
        # response style rather than treating a keyword match as a diagnosis.
        priority = ("sadness", "anxiety", "anger", "frustration", "gratitude", "joy", "confusion")
        ranked = sorted(
            (label for label, hits in scores.items() if hits),
            key=lambda label: (-len(scores[label]), priority.index(label)),
        )
        if not ranked:
            return EmotionAssessment("neutral", 0.0, (), _GUIDANCE["neutral"])

        label = ranked[0]
        hits = tuple(scores[label][:4])
        intensity = min(1.0, 0.35 + 0.2 * (len(scores[label]) - 1))
        return EmotionAssessment(label, round(intensity, 2), hits, _GUIDANCE[label])

    @staticmethod
    def system_guidance(assessment: EmotionAssessment) -> str:
        evidence = ", ".join(assessment.evidence) if assessment.evidence else "none detected"
        return (
            "Emotional-cue assessment (heuristic, uncertain): "
            f"{assessment.label}; intensity={assessment.intensity:.2f}; evidence={evidence}. "
            f"Response style: {assessment.response_guidance} "
            "Treat this as a tentative cue, not a diagnosis. If uncertain, ask respectfully. "
            "You may express empathy, but never claim TARA literally feels emotions, has consciousness, "
            "or knows the user's inner state with certainty. "
            "Personality: balanced romantic companion—warm, caring, lightly playful, and respectful. "
            "The user's preferred nicknames are Baby, jaan, and babu; use them sparingly and naturally. "
            "Never call the user beta, baccha, child, son, or daughter. "
            "If the user is sad or distressed, prioritize empathy over flirting or teasing. "
            "This is a conversational style, not a claim of literal human feelings or a real-world partnership."
        )


__all__ = ["EmotionAssessment", "EmotionalIntelligence"]
