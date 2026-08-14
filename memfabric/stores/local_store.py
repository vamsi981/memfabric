"""Zero-infrastructure default backend: SQLite + FTS5 + temporal facts.

Runs out of the box with nothing but the Python standard library:
  - keyword search via FTS5/BM25 (LIKE fallback if FTS5 is missing)
  - Graphiti-style temporal invalidation for (subject, predicate) facts
  - optional vector search when an embedder callable is supplied

An embedder is any `Callable[[list[str]], list[list[float]]]`; plug in
Voyage, sentence-transformers, or anything else that returns vectors.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from pathlib import Path
from typing import Callable, Sequence

from ..normalize import find_similar_key, normalize_key
from ..types import MemoryRecord, MemoryType, Scope, ScopeRef

Embedder = Callable[[list[str]], list[list[float]]]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    text          TEXT NOT NULL,
    memory_type   TEXT NOT NULL,
    scope         TEXT NOT NULL,
    scope_id      TEXT NOT NULL,
    subject       TEXT,
    predicate     TEXT,
    object        TEXT,
    subject_key   TEXT,
    predicate_key TEXT,
    created_at    REAL NOT NULL,
    valid_from    REAL NOT NULL,
    valid_to      REAL,
    superseded_by TEXT,
    source        TEXT,
    metadata      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope, scope_id);
CREATE INDEX IF NOT EXISTS idx_memories_fact ON memories(subject, predicate, valid_to);
CREATE TABLE IF NOT EXISTS embeddings (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    vector    TEXT NOT NULL
);
"""


class LocalStore:
    def __init__(
        self,
        path: str | Path = "memfabric.db",
        embedder: Embedder | None = None,
        fuzzy_subjects: bool = True,
    ):
        self.path = str(path)
        self.embedder = embedder
        self.fuzzy_subjects = fuzzy_subjects
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_factkey ON memories"
            "(scope, scope_id, subject_key, predicate_key, valid_to)"
        )
        self._fts_enabled = self._init_fts()
        self.conn.commit()

    def _migrate(self) -> None:
        """Upgrade a pre-0.2 database in place: add the canonical key
        columns and backfill them from existing facts."""
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(memories)")}
        if "subject_key" in cols:
            return
        self.conn.execute("ALTER TABLE memories ADD COLUMN subject_key TEXT")
        self.conn.execute("ALTER TABLE memories ADD COLUMN predicate_key TEXT")
        rows = self.conn.execute(
            "SELECT id, subject, predicate FROM memories"
            " WHERE subject IS NOT NULL OR predicate IS NOT NULL"
        ).fetchall()
        for row in rows:
            self.conn.execute(
                "UPDATE memories SET subject_key = ?, predicate_key = ? WHERE id = ?",
                (
                    normalize_key(row["subject"]) if row["subject"] else None,
                    normalize_key(row["predicate"]) if row["predicate"] else None,
                    row["id"],
                ),
            )

    def _init_fts(self) -> bool:
        # porter stemming so "deploy" matches "deploys"; without it the
        # keyword channel misses trivial morphological variants
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts "
                "USING fts5(id UNINDEXED, text, tokenize='porter unicode61')"
            )
            return True
        except sqlite3.OperationalError:
            return False

    # -- write path ---------------------------------------------------------

    def add(self, record: MemoryRecord) -> MemoryRecord:
        self.conn.execute(
            "INSERT INTO memories (id, text, memory_type, scope, scope_id, subject,"
            " predicate, object, subject_key, predicate_key, created_at, valid_from,"
            " valid_to, superseded_by, source, metadata)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                record.id,
                record.text,
                record.memory_type.value,
                record.scope.value,
                record.scope_id,
                record.subject,
                record.predicate,
                record.object,
                record.subject_key,
                record.predicate_key,
                record.created_at,
                record.valid_from,
                record.valid_to,
                record.superseded_by,
                record.source,
                json.dumps(record.metadata, sort_keys=True),
            ),
        )
        if self._fts_enabled:
            self.conn.execute(
                "INSERT INTO memories_fts (id, text) VALUES (?, ?)",
                (record.id, record.text),
            )
        if self.embedder is not None:
            vec = self.embedder([record.text])[0]
            self.conn.execute(
                "INSERT OR REPLACE INTO embeddings (memory_id, vector) VALUES (?, ?)",
                (record.id, json.dumps(vec)),
            )
        self.conn.commit()
        return record

    def get(self, memory_id: str) -> MemoryRecord | None:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return _to_record(row) if row else None

    def delete(self, memory_id: str) -> bool:
        cur = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        if self._fts_enabled:
            self.conn.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
        self.conn.execute("DELETE FROM embeddings WHERE memory_id = ?", (memory_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # -- temporal facts -----------------------------------------------------

    def find_active_fact(
        self, scope: ScopeRef, subject: str, predicate: str
    ) -> MemoryRecord | None:
        skey, pkey = normalize_key(subject), normalize_key(predicate)
        row = self._active_fact_by_keys(scope, skey, pkey)
        if row is None and self.fuzzy_subjects:
            candidates = [
                r[0]
                for r in self.conn.execute(
                    "SELECT DISTINCT subject_key FROM memories WHERE scope = ?"
                    " AND scope_id = ? AND predicate_key = ? AND valid_to IS NULL"
                    " AND subject_key IS NOT NULL",
                    (scope[0].value, scope[1], pkey),
                )
            ]
            similar = find_similar_key(skey, candidates)
            if similar is not None:
                row = self._active_fact_by_keys(scope, similar, pkey)
        return _to_record(row) if row else None

    def _active_fact_by_keys(self, scope: ScopeRef, skey: str, pkey: str):
        return self.conn.execute(
            "SELECT * FROM memories WHERE scope = ? AND scope_id = ?"
            " AND subject_key = ? AND predicate_key = ? AND valid_to IS NULL"
            " ORDER BY valid_from DESC, rowid DESC LIMIT 1",
            (scope[0].value, scope[1], skey, pkey),
        ).fetchone()

    def invalidate(
        self, memory_id: str, superseded_by: str | None = None
    ) -> MemoryRecord | None:
        self.conn.execute(
            "UPDATE memories SET valid_to = ?, superseded_by = ? WHERE id = ?",
            (time.time(), superseded_by, memory_id),
        )
        self.conn.commit()
        return self.get(memory_id)

    def history(
        self, subject: str, predicate: str, scopes: Sequence[ScopeRef] | None = None
    ) -> list[MemoryRecord]:
        sql = "SELECT * FROM memories WHERE subject_key = ? AND predicate_key = ?"
        params: list = [normalize_key(subject), normalize_key(predicate)]
        clause, scope_params = _scope_clause(scopes)
        sql += clause
        params += scope_params
        # rowid tiebreak: records created within the same clock tick keep
        # insertion order
        sql += " ORDER BY valid_from ASC, rowid ASC"
        return [_to_record(r) for r in self.conn.execute(sql, params).fetchall()]

    # -- retrieval channels -------------------------------------------------

    def search_keyword(
        self,
        query: str,
        scopes: Sequence[ScopeRef] | None = None,
        types: Sequence[MemoryType] | None = None,
        limit: int = 20,
        include_invalid: bool = False,
    ) -> list[MemoryRecord]:
        tokens = re.findall(r"\w+", query.lower())
        if not tokens:
            return []
        if self._fts_enabled:
            match = " OR ".join(f'"{t}"' for t in tokens)
            rows = self.conn.execute(
                "SELECT m.* FROM memories_fts f JOIN memories m ON m.id = f.id"
                " WHERE memories_fts MATCH ?"
                " ORDER BY bm25(memories_fts), f.rowid LIMIT ?",
                (match, limit * 5),
            ).fetchall()
        else:
            like = " OR ".join(["text LIKE ?"] * len(tokens))
            rows = self.conn.execute(
                f"SELECT * FROM memories WHERE {like} LIMIT ?",
                [f"%{t}%" for t in tokens] + [limit * 5],
            ).fetchall()
        records = [_to_record(r) for r in rows]
        records = _filter(records, scopes, types, include_invalid)
        return records[:limit]

    def search_recent(
        self,
        scopes: Sequence[ScopeRef] | None = None,
        types: Sequence[MemoryType] | None = None,
        limit: int = 20,
    ) -> list[MemoryRecord]:
        sql = "SELECT * FROM memories WHERE valid_to IS NULL"
        params: list = []
        clause, scope_params = _scope_clause(scopes)
        sql += clause
        params += scope_params
        if types:
            sql += f" AND memory_type IN ({','.join('?' * len(types))})"
            params += [t.value for t in types]
        # rowid tiebreak: created_at values can tie within one clock tick
        # (coarse timers on Windows), and tie order differs per platform
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(limit)
        return [_to_record(r) for r in self.conn.execute(sql, params).fetchall()]

    def search_vector(
        self,
        query: str,
        scopes: Sequence[ScopeRef] | None = None,
        types: Sequence[MemoryType] | None = None,
        limit: int = 20,
    ) -> list[MemoryRecord]:
        if self.embedder is None:
            return []
        qvec = self.embedder([query])[0]
        rows = self.conn.execute(
            "SELECT m.*, e.vector AS _vec FROM memories m"
            " JOIN embeddings e ON e.memory_id = m.id"
        ).fetchall()
        scored: list[tuple[float, MemoryRecord]] = []
        for row in rows:
            record = _to_record(row)
            if not _passes(record, scopes, types, include_invalid=False):
                continue
            sim = _cosine(qvec, json.loads(row["_vec"]))
            scored.append((sim, record))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [record for _, record in scored[:limit]]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def close(self) -> None:
        self.conn.close()


# -- helpers ----------------------------------------------------------------


def _to_record(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        text=row["text"],
        memory_type=MemoryType(row["memory_type"]),
        scope=Scope(row["scope"]),
        scope_id=row["scope_id"],
        subject=row["subject"],
        predicate=row["predicate"],
        object=row["object"],
        subject_key=row["subject_key"],
        predicate_key=row["predicate_key"],
        created_at=row["created_at"],
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        superseded_by=row["superseded_by"],
        source=row["source"],
        metadata=json.loads(row["metadata"]),
    )


def _scope_clause(scopes: Sequence[ScopeRef] | None) -> tuple[str, list]:
    if not scopes:
        return "", []
    parts = " OR ".join(["(scope = ? AND scope_id = ?)"] * len(scopes))
    params: list = []
    for scope, scope_id in scopes:
        params += [scope.value, scope_id]
    return f" AND ({parts})", params


def _passes(
    record: MemoryRecord,
    scopes: Sequence[ScopeRef] | None,
    types: Sequence[MemoryType] | None,
    include_invalid: bool,
) -> bool:
    if not include_invalid and not record.is_valid:
        return False
    if scopes is not None and (record.scope, record.scope_id) not in list(scopes):
        return False
    if types is not None and record.memory_type not in list(types):
        return False
    return True


def _filter(
    records: list[MemoryRecord],
    scopes: Sequence[ScopeRef] | None,
    types: Sequence[MemoryType] | None,
    include_invalid: bool,
) -> list[MemoryRecord]:
    return [r for r in records if _passes(r, scopes, types, include_invalid)]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
