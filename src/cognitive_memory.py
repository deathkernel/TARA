"""Schema-flexible persistent cognitive memory for TARA.

The store keeps a stable envelope around arbitrary JSON knowledge. New
knowledge types do not require a database migration: callers can add any
``kind`` and any JSON-serializable ``content`` while provenance, verification,
relationships, and event history remain queryable through stable fields.

This is deliberately an inspectable foundation. It does not claim semantic
understanding or autonomous learning by itself; higher-level learning loops
can use it as durable external memory.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class CognitiveMemory:
    """Persistent, schema-flexible memory backed by SQLite."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._initialize()

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                domain TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                confidence REAL NOT NULL,
                verification TEXT NOT NULL,
                provenance TEXT NOT NULL,
                content TEXT NOT NULL,
                attributes TEXT NOT NULL,
                fingerprint TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS relationships (
                source_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                target_id TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (source_id, relation, target_id),
                FOREIGN KEY (source_id) REFERENCES knowledge(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES knowledge(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledge_id TEXT,
                event_type TEXT NOT NULL,
                created_at REAL NOT NULL,
                payload TEXT NOT NULL,
                FOREIGN KEY (knowledge_id) REFERENCES knowledge(id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_knowledge_kind ON knowledge(kind);
            CREATE INDEX IF NOT EXISTS idx_knowledge_domain ON knowledge(domain);
            CREATE INDEX IF NOT EXISTS idx_knowledge_verification ON knowledge(verification);
            CREATE INDEX IF NOT EXISTS idx_relationship_source ON relationships(source_id);
            CREATE INDEX IF NOT EXISTS idx_relationship_target ON relationships(target_id);
            CREATE INDEX IF NOT EXISTS idx_events_knowledge ON events(knowledge_id);
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "CognitiveMemory":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def remember(
        self,
        *,
        kind: str,
        content: Any,
        domain: str | None = None,
        confidence: float = 0.0,
        verification: str = "unverified",
        provenance: Any = None,
        attributes: dict[str, Any] | None = None,
        knowledge_id: str | None = None,
    ) -> str:
        """Store arbitrary JSON knowledge using a stable metadata envelope."""
        if not kind or not isinstance(kind, str):
            raise ValueError("kind must be a non-empty string")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if verification not in {"unverified", "verified", "rejected", "superseded"}:
            raise ValueError("invalid verification state")
        if attributes is not None and not isinstance(attributes, dict):
            raise TypeError("attributes must be a JSON object")

        now = time.time()
        item_id = knowledge_id or str(uuid.uuid4())
        provenance_value = {} if provenance is None else provenance
        attributes_value = {} if attributes is None else attributes
        fingerprint = _fingerprint(
            {
                "kind": kind,
                "domain": domain,
                "content": content,
                "attributes": attributes_value,
            }
        )
        payload = (
            item_id,
            kind,
            domain,
            now,
            now,
            float(confidence),
            verification,
            _json(provenance_value),
            _json(content),
            _json(attributes_value),
            fingerprint,
        )
        self._connection.execute(
            """
            INSERT INTO knowledge
            (id, kind, domain, created_at, updated_at, confidence, verification,
             provenance, content, attributes, fingerprint)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        self._record_event(item_id, "remembered", {"fingerprint": fingerprint})
        self._connection.commit()
        return item_id

    def update(self, knowledge_id: str, **changes: Any) -> None:
        """Update envelope/content fields without imposing a schema on content."""
        allowed = {
            "kind", "domain", "confidence", "verification", "provenance",
            "content", "attributes",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"unsupported fields: {sorted(unknown)}")
        current = self.get(knowledge_id)
        if current is None:
            raise KeyError(knowledge_id)

        merged = dict(current)
        merged.update(changes)
        if not 0.0 <= float(merged["confidence"]) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if merged["verification"] not in {"unverified", "verified", "rejected", "superseded"}:
            raise ValueError("invalid verification state")
        if not isinstance(merged["attributes"], dict):
            raise TypeError("attributes must be a JSON object")

        fingerprint = _fingerprint(
            {
                "kind": merged["kind"],
                "domain": merged["domain"],
                "content": merged["content"],
                "attributes": merged["attributes"],
            }
        )
        self._connection.execute(
            """
            UPDATE knowledge SET kind=?, domain=?, updated_at=?, confidence=?,
                verification=?, provenance=?, content=?, attributes=?, fingerprint=?
            WHERE id=?
            """,
            (
                merged["kind"], merged["domain"], time.time(), float(merged["confidence"]),
                merged["verification"], _json(merged["provenance"]),
                _json(merged["content"]), _json(merged["attributes"]), fingerprint,
                knowledge_id,
            ),
        )
        self._record_event(knowledge_id, "updated", {"fingerprint": fingerprint})
        self._connection.commit()

    def get(self, knowledge_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT * FROM knowledge WHERE id=?", (knowledge_id,)
        ).fetchone()
        return self._decode_knowledge(row) if row else None

    def search(
        self,
        *,
        kind: str | None = None,
        domain: str | None = None,
        verification: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        clauses: list[str] = []
        params: list[Any] = []
        for field, value in (("kind", kind), ("domain", domain), ("verification", verification)):
            if value is not None:
                clauses.append(f"{field}=?")
                params.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._connection.execute(
            "SELECT * FROM knowledge" + where + " ORDER BY updated_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [self._decode_knowledge(row) for row in rows]

    def link(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.get(source_id) is None or self.get(target_id) is None:
            raise KeyError("both relationship endpoints must exist")
        self._connection.execute(
            """
            INSERT OR REPLACE INTO relationships
            (source_id, relation, target_id, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (source_id, relation, target_id, _json(metadata or {}), time.time()),
        )
        self._record_event(source_id, "relationship_added", {
            "relation": relation, "target_id": target_id,
        })
        self._connection.commit()

    def related(
        self,
        knowledge_id: str,
        *,
        relation: str | None = None,
    ) -> list[dict[str, Any]]:
        clause = "AND r.relation=?" if relation is not None else ""
        params: Iterable[Any] = (knowledge_id, relation) if relation is not None else (knowledge_id,)
        rows = self._connection.execute(
            f"""
            SELECT r.relation, r.metadata, r.created_at, k.*
            FROM relationships r
            JOIN knowledge k ON k.id = r.target_id
            WHERE r.source_id=? {clause}
            ORDER BY r.created_at ASC
            """,
            tuple(params),
        ).fetchall()
        results = []
        for row in rows:
            item = self._decode_knowledge(row)
            item["relation"] = row["relation"]
            item["relationship_metadata"] = json.loads(row["metadata"])
            item["relationship_created_at"] = row["created_at"]
            results.append(item)
        return results

    def events(self, knowledge_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        clause = " WHERE knowledge_id=?" if knowledge_id is not None else ""
        params: tuple[Any, ...] = (knowledge_id, limit) if knowledge_id is not None else (limit,)
        rows = self._connection.execute(
            "SELECT * FROM events" + clause + " ORDER BY event_id DESC LIMIT ?",
            params,
        ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "knowledge_id": row["knowledge_id"],
                "event_type": row["event_type"],
                "created_at": row["created_at"],
                "payload": json.loads(row["payload"]),
            }
            for row in rows
        ]

    def count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0])

    def _record_event(self, knowledge_id: str | None, event_type: str, payload: Any) -> None:
        self._connection.execute(
            "INSERT INTO events (knowledge_id, event_type, created_at, payload) VALUES (?, ?, ?, ?)",
            (knowledge_id, event_type, time.time(), _json(payload)),
        )

    @staticmethod
    def _decode_knowledge(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "kind": row["kind"],
            "domain": row["domain"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "confidence": row["confidence"],
            "verification": row["verification"],
            "provenance": json.loads(row["provenance"]),
            "content": json.loads(row["content"]),
            "attributes": json.loads(row["attributes"]),
            "fingerprint": row["fingerprint"],
        }
