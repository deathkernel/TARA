"""Polyglot algorithm planning, candidates, execution, and improvement."""

from .candidate import PolyglotCandidate
from .execution import PolyglotExecutionError, PolyglotExecutor
from .generation import ModelCandidateGenerator, build_candidate_prompt
from .languages import LanguageSpec, default_languages, select_language
from .model_adapter import TARAAlgorithmModel
from .parser import CandidateParseError, parse_candidate
from .result import ExecutionResult

__all__ = [
    "CandidateParseError",
    "ExecutionResult",
    "LanguageSpec",
    "ModelCandidateGenerator",
    "PolyglotCandidate",
    "PolyglotExecutionError",
    "PolyglotExecutor",
    "TARAAlgorithmModel",
    "build_candidate_prompt",
    "default_languages",
    "parse_candidate",
    "select_language",
]
