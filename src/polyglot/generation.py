"""Prompt construction and model-output generation for polyglot candidates."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .candidate import PolyglotCandidate
from .languages import LanguageSpec, default_languages, select_language
from .parser import CandidateParseError, parse_candidate
from .problems import ProblemSpec, get_problem

GenerateText = Callable[..., str]


def build_candidate_prompt(problem: str, language: str, feedback: Iterable[str] = ()) -> str:
    """Build a deterministic prompt containing the executable I/O contract."""
    spec = get_problem(problem)
    feedback_text = "\n".join(f"- {item}" for item in feedback if item.strip())
    if not feedback_text:
        feedback_text = "- No previous feedback; produce a clean first attempt."
    language_spec = next((item for item in default_languages() if item.name == language), None)
    extension = language_spec.file_extension if language_spec else ".txt"
    class_hint = " The public class must be named candidate." if language == "java" else ""
    examples = "\n".join(
        f"Example {index}: INPUT={case.stdin!r} OUTPUT={case.expected_stdout!r}"
        for index, case in enumerate(spec.tests, start=1)
    )
    return (
        "TARA ALGORITHM CANDIDATE TASK\n"
        f"Problem: {spec.name}\n"
        f"Description: {spec.description}\n"
        f"Target language: {language}\n"
        f"Required file extension: {extension}\n"
        "I/O examples:\n"
        f"{examples}\n"
        "Write one complete standalone program. Follow the input/output contract exactly. "
        "Handle general valid inputs, not only the examples. Do not explain the solution inside the code. "
        "Do not use external packages."
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
        languages: tuple[LanguageSpec, ...] | None = None,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not temperatures or any(value <= 0 for value in temperatures):
            raise ValueError("temperatures must contain positive values")
        if languages is not None and not languages:
            raise ValueError("languages must not be empty")
        self.generate_text = generate_text
        self.max_new_tokens = max_new_tokens
        self.temperatures = temperatures
        self.languages = languages

    def __call__(self, problem: str, feedback_text: str, feedback: tuple[str, ...], count: int) -> Iterable[PolyglotCandidate]:
        if count <= 0:
            return ()
        requested = self.languages or (select_language(problem),)
        feedback_items = feedback or tuple(line.strip("- ") for line in feedback_text.splitlines() if line.strip())
        candidates: list[PolyglotCandidate] = []
        seen: set[tuple[str, str]] = set()
        for index in range(count):
            language = requested[index % len(requested)].name
            temperature = self.temperatures[index % len(self.temperatures)]
            prompt = build_candidate_prompt(problem, language, feedback_items)
            try:
                output = self.generate_text(prompt, max_new_tokens=self.max_new_tokens, temperature=temperature)
                candidate = parse_candidate(output, problem)
            except (CandidateParseError, ValueError, TypeError):
                continue
            if candidate.language != language:
                continue
            key = (candidate.language, candidate.source)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
        return tuple(candidates)
