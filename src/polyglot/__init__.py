"""Polyglot algorithm planning, candidates, execution, and improvement."""

from .archive import CandidateArchive
from .benchmark import BenchmarkResult, PolyglotBenchmark, rank_benchmarks
from .candidate import PolyglotCandidate
from .execution import PolyglotExecutionError, PolyglotExecutor
from .generation import ModelCandidateGenerator, build_candidate_prompt
from .languages import LanguageSpec, default_languages, select_language
from .model_adapter import TARAAlgorithmModel
from .parser import CandidateParseError, parse_candidate
from .result import ExecutionResult
from .self_improvement import ImprovementResult, SelfImprovementEngine

__all__ = [
    "BenchmarkResult",
    "CandidateArchive",
    "CandidateParseError",
    "ExecutionResult",
    "ImprovementResult",
    "LanguageSpec",
    "ModelCandidateGenerator",
    "PolyglotBenchmark",
    "PolyglotCandidate",
    "PolyglotExecutionError",
    "PolyglotExecutor",
    "SelfImprovementEngine",
    "TARAAlgorithmModel",
    "build_candidate_prompt",
    "default_languages",
    "parse_candidate",
    "rank_benchmarks",
    "select_language",
]
