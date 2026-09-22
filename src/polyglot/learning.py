"""Offline learning orchestration for verified algorithm knowledge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .archive import CandidateArchive
from .knowledge import KnowledgeExtractor


@dataclass(frozen=True)
class LearningExport:
    """Description of a dataset export; no model training is performed here."""

    path: str
    records: int


class VerifiedKnowledgeLearner:
    """Build a clean training corpus from TARA's verified research memory.

    This layer intentionally stops at dataset creation. Model training is an
    explicit offline operation so unverified generated code cannot silently
    become training data.
    """

    def __init__(self, archive: CandidateArchive, extractor: KnowledgeExtractor | None = None) -> None:
        self.archive = archive
        self.extractor = extractor or KnowledgeExtractor(verified_only=True)

    def export(self, path: str | Path, problem: str | None = None) -> LearningExport:
        destination = Path(path)
        count = self.extractor.export_archive(self.archive, destination, problem)
        return LearningExport(path=str(destination), records=count)
