"""Prompt construction and model-output generation for polyglot candidates."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .candidate import PolyglotCandidate
from .languages import default_languages, select_language
from .parser import CandidateParseError, parse_candidate

GenerateText = Callable[..., str]


def build_candidate_prompt(problem: str, language: str, feedback: Iterable[str] = ()) -> str:
    """Build a deterministic instruction for one executable candidate."""
    feedback_text = "\n".join(f"- {item}" for item in feedback if item.strip())
    if not feedback_text:
        feedback_text = "- No previous feedback; produce a clean first attempt."
    spec = next((item for item in default_languages() if item.name == language), None)
    extension = spec.file_extension if spec else ".txt"
    class_hint = " If using Java, the public class must be named candidate." if language == "java" else ""
    return (
        "TARA ALGORITHM CANDIDATE TASK\n"
        f"Problem: {problem}\n"
        f"Target language: {language}\n"
        f"Required file extension: {extension}\n"
        "Write one complete standalone program. Follow the problem input/output contract exactly. "
        "Do not explain the solution inside the code. Do not use external packages."
        f"{class_hint}\n"
        "Return exactly one fenced code block and put the language after the opening backticks.\n"
        "Format:\n"
        f"```{language}\n<complete program>\n```\n"
        "Previous verification feedback:\n"
        f"{feedback_text}\n"
    )


class ModelCandidateGenerator:
    """Adapt TARA's text-generation callable to SelfImprovementEngine."""

    def __init__(
        self,
        generate_text: GenerateText,
        *,
        max_new_tokens: int = 512,
        temperatures: tuple[float, ...] = (0.65, 0.85, 1.05),
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not temperatures or any(value <= 0 for value in temperatures):
            raise ValueError("temperatures must contain positive values")
        self.generate_text = generate_text
        self.max_new_tokens = max_new_tokens
        self.temperatures = temperatures

    def __call__(self, problem: str, feedback_text: str, feedback: tuple[str, ...], count: int) -> Iterable[PolyglotCandidate]:
        if count <= 0:
            return ()
        requested_language = select_language(problem).name
        feedback_items = feedback or tuple(line.strip("- ") for line in feedback_text.splitlines() if line.strip())
        candidates: list[PolyglotCandidate] = []
        seen: set[tuple[str, str]] = set()
        for index in range(count):
            temperature = self.temperatures[index % len(self.temperatures)]
            prompt = build_candidate_prompt(problem, requested_language, feedback_items)
            try:
                output = self.generate_text(prompt, max_new_tokens=self.max_new_tokens, temperature=temperature)
                candidate = parse_candidate(output, problem)
            except (CandidateParseError, ValueError, TypeError) as exc:
                # Parsing failures are surfaced as feedback on the next round,
                # but malformed model output is never sent to an executor.
                continue
            key = (candidate.language, candidate.source)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
        return tuple(candidates)
