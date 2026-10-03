from __future__ import annotations

import json

from src.upstream.registry import UpstreamRegistry


def test_manifest_contains_requested_ecosystem(tmp_path):
    manifest = tmp_path / "manifest.json"
    lock = tmp_path / "lock.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "sources": [
            {
                "id": "tensorflow",
                "repo": "tensorflow/tensorflow",
                "ref": "master",
                "kind": "runtime",
                "capabilities": ["tensor-runtime"],
                "adapter": "tensorflow",
                "package": "tensorflow",
                "optional": True,
            },
            {
                "id": "gpt-oss",
                "repo": "openai/gpt-oss",
                "ref": "main",
                "kind": "model",
                "capabilities": ["reasoning"],
                "adapter": "gpt-oss",
                "package": None,
                "optional": True,
            },
        ],
    }), encoding="utf-8")
    registry = UpstreamRegistry(manifest, lock)
    assert {item.id for item in registry.sources} == {"tensorflow", "gpt-oss"}


def test_drift_detects_movement(tmp_path):
    manifest = tmp_path / "manifest.json"
    lock = tmp_path / "lock.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "sources": [{
            "id": "x",
            "repo": "owner/repo",
            "ref": "main",
            "kind": "runtime",
            "capabilities": [],
            "adapter": "x",
            "package": None,
        }],
    }), encoding="utf-8")
    lock.write_text(json.dumps({
        "schema_version": 1,
        "sources": {"x": {"sha": "old"}},
    }), encoding="utf-8")
    registry = UpstreamRegistry(manifest, lock)

    source = registry.sources[0]
    snapshot = type("Snapshot", (), {
        "source": source,
        "sha": "new",
        "message": "changed",
        "url": "https://github.com/owner/repo/commit/new",
    })()
    changes = registry.drift((snapshot,))
    assert changes[0]["previous_sha"] == "old"
    assert changes[0]["current_sha"] == "new"


def test_capability_map_is_deterministic(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "sources": [{
            "id": "x",
            "repo": "owner/repo",
            "ref": "main",
            "kind": "runtime",
            "capabilities": ["a", "b"],
            "adapter": "x",
            "package": None,
        }],
    }), encoding="utf-8")
    registry = UpstreamRegistry(manifest, tmp_path / "missing.json")
    assert registry.capability_map() == {"x": ("a", "b")}
