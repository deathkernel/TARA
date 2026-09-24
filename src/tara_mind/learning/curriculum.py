"""Curriculum manifest helpers for TARA Baby."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_curriculum(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("program") != "TARA Baby":
        raise ValueError("curriculum is not for TARA Baby")
    stages = data.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("curriculum must contain at least one stage")
    return data


def stage_ids(manifest: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(stage["id"]) for stage in manifest["stages"])


def datasets_for_stage(manifest: dict[str, Any], stage_id: str) -> list[dict[str, Any]]:
    for stage in manifest["stages"]:
        if stage["id"] == stage_id:
            return list(stage.get("datasets", []))
    raise KeyError(f"unknown curriculum stage: {stage_id}")
