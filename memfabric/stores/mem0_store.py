"""Adapter backend for Mem0 (https://github.com/mem0ai/mem0).

Requires `pip install mem0ai` (extra: `pip install -e .[mem0]`) plus whatever
LLM/embedder/vector-store config Mem0 itself needs. Mem0 owns extraction and
dedup on its side, so MemoryFabric passes raw text through and skips its own
extraction when this backend is active.

Scope mapping: Mem0's user_id carries our "scope:scope_id" key, so every
Fabric scope (user/team/project/org/...) gets an isolated Mem0 namespace and
permission-aware recall keeps working.

Temporal operations (find_active_fact/invalidate/history) are not supported
here; Mem0 manages fact updates internally. MemoryFabric catches the
NotImplementedError and skips supersede logic for this backend.
"""

from __future__ import annotations

from typing import Sequence

from ..types import MemoryRecord, MemoryType, Scope, ScopeRef


def _ns(scope: Scope, scope_id: str) -> str:
    return f"{scope.value}:{scope_id}"


class Mem0Store:
    OWNS_EXTRACTION = True  # Mem0 runs its own extraction/dedup pipeline

    def __init__(self, config: dict | None = None):
        try:
            from mem0 import Memory
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Mem0Store requires the 'mem0ai' package: pip install mem0ai"
            ) from exc
        self.memory = Memory.from_config(config) if config else Memory()

    def add(self, record: MemoryRecord) -> MemoryRecord:
        result = self.memory.add(
            record.text,
            user_id=_ns(record.scope, record.scope_id),
            metadata={
                "fabric_id": record.id,
                "memory_type": record.memory_type.value,
                "source": record.source,
                **record.metadata,
            },
        )
        record.metadata["mem0_result"] = result
        return record

    def get(self, memory_id: str) -> MemoryRecord | None:
        raise NotImplementedError("Mem0Store does not support get-by-fabric-id")

    def delete(self, memory_id: str) -> bool:
        self.memory.delete(memory_id)
        return True

    def search_keyword(
        self,
        query: str,
        scopes: Sequence[ScopeRef] | None = None,
        types: Sequence[MemoryType] | None = None,
        limit: int = 20,
        include_invalid: bool = False,
    ) -> list[MemoryRecord]:
        # Mem0 search is already hybrid (vector + graph); expose it on the
        # keyword channel and leave the vector channel empty to avoid double
        # counting in fusion.
        records: list[MemoryRecord] = []
        for scope, scope_id in scopes or [(Scope.USER, "default")]:
            result = self.memory.search(
                query, user_id=_ns(scope, scope_id), limit=limit
            )
            hits = result.get("results", result) if isinstance(result, dict) else result
            for hit in hits:
                records.append(
                    MemoryRecord(
                        id=str(hit.get("id", "")),
                        text=hit.get("memory", hit.get("text", "")),
                        memory_type=MemoryType.SEMANTIC,
                        scope=scope,
                        scope_id=scope_id,
                        metadata={"mem0_score": hit.get("score")},
                    )
                )
        return records[:limit]

    def search_recent(
        self,
        scopes: Sequence[ScopeRef] | None = None,
        types: Sequence[MemoryType] | None = None,
        limit: int = 20,
    ) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for scope, scope_id in scopes or [(Scope.USER, "default")]:
            result = self.memory.get_all(user_id=_ns(scope, scope_id), limit=limit)
            hits = result.get("results", result) if isinstance(result, dict) else result
            for hit in hits:
                records.append(
                    MemoryRecord(
                        id=str(hit.get("id", "")),
                        text=hit.get("memory", hit.get("text", "")),
                        scope=scope,
                        scope_id=scope_id,
                    )
                )
        return records[:limit]

    def search_vector(self, query, scopes=None, types=None, limit=20):
        return []  # folded into search_keyword (Mem0 search is already hybrid)

    def find_active_fact(self, scope, subject, predicate):
        raise NotImplementedError("Mem0 manages fact updates internally")

    def invalidate(self, memory_id, superseded_by=None):
        raise NotImplementedError("Mem0 manages fact updates internally")

    def history(self, subject, predicate, scopes=None):
        raise NotImplementedError("Mem0 does not expose temporal fact history")
