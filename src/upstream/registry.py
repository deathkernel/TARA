"""Registry and update coordinator for TARA's external open-source stack."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "config" / "upstream_integrations.json"
DEFAULT_LOCK = ROOT / "config" / "upstream_lock.json"


@dataclass(frozen=True)
class UpstreamSource:
    id: str
    repo: str
    ref: str
    kind: str
    capabilities: tuple[str, ...]
    adapter: str
    package: str | None
    optional: bool = True


@dataclass(frozen=True)
class UpstreamSnapshot:
    source: UpstreamSource
    sha: str
    message: str
    url: str


class UpstreamRegistry:
    """Read TARA's upstream manifest and resolve current GitHub refs."""

    def __init__(self, manifest: Path | str = DEFAULT_MANIFEST, lock: Path | str = DEFAULT_LOCK):
        self.manifest_path = Path(manifest)
        self.lock_path = Path(lock)
        self.sources = self._load_manifest()

    def _load_manifest(self) -> tuple[UpstreamSource, ...]:
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError("unsupported upstream manifest schema")
        sources = []
        seen = set()
        for raw in data.get("sources", []):
            source = UpstreamSource(
                id=str(raw["id"]),
                repo=str(raw["repo"]),
                ref=str(raw["ref"]),
                kind=str(raw["kind"]),
                capabilities=tuple(raw.get("capabilities", [])),
                adapter=str(raw["adapter"]),
                package=raw.get("package"),
                optional=bool(raw.get("optional", True)),
            )
            if source.id in seen:
                raise ValueError(f"duplicate upstream id: {source.id}")
            seen.add(source.id)
            sources.append(source)
        if not sources:
            raise ValueError("upstream manifest is empty")
        return tuple(sources)

    @staticmethod
    def _github_commit(repo: str, ref: str) -> dict[str, Any]:
        url = f"https://api.github.com/repos/{repo}/commits/{ref}"
        request = Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "TARA-upstream-sync/1.0",
            },
        )
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))

    def resolve(self, source: UpstreamSource) -> UpstreamSnapshot:
        payload = self._github_commit(source.repo, source.ref)
        commit = payload.get("commit", {})
        return UpstreamSnapshot(
            source=source,
            sha=str(payload["sha"]),
            message=str(commit.get("message", "")).splitlines()[0],
            url=str(payload.get("html_url") or f"https://github.com/{source.repo}/commit/{payload['sha']}"),
        )

    def resolve_all(self) -> tuple[UpstreamSnapshot, ...]:
        return tuple(self.resolve(source) for source in self.sources)

    def read_lock(self) -> dict[str, Any]:
        if not self.lock_path.exists():
            return {"schema_version": 1, "generated_at": None, "sources": {}}
        return json.loads(self.lock_path.read_text(encoding="utf-8"))

    def drift(self, snapshots: tuple[UpstreamSnapshot, ...] | None = None) -> list[dict[str, str]]:
        snapshots = snapshots or self.resolve_all()
        locked = self.read_lock().get("sources", {})
        changes = []
        for snapshot in snapshots:
            previous = locked.get(snapshot.source.id, {})
            if previous.get("sha") != snapshot.sha:
                changes.append({
                    "id": snapshot.source.id,
                    "repo": snapshot.source.repo,
                    "previous_sha": str(previous.get("sha") or ""),
                    "current_sha": snapshot.sha,
                    "message": snapshot.message,
                    "url": snapshot.url,
                })
        return changes

    def write_lock(self, snapshots: tuple[UpstreamSnapshot, ...]) -> None:
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sources": {
                snapshot.source.id: {
                    "repo": snapshot.source.repo,
                    "ref": snapshot.source.ref,
                    "sha": snapshot.sha,
                    "message": snapshot.message,
                    "url": snapshot.url,
                }
                for snapshot in snapshots
            },
        }
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def capability_map(self) -> dict[str, tuple[str, ...]]:
        return {source.id: source.capabilities for source in self.sources}
