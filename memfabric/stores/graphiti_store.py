"""Adapter backend for Graphiti (https://github.com/getzep/graphiti).

Requires `pip install graphiti-core` (extra: `pip install -e .[graphiti]`) and
a running Neo4j (or FalkorDB) instance. Graphiti owns the temporal knowledge
graph: episode ingestion, entity/relationship extraction, and edge
invalidation all happen on its side.

Verify method names against the installed graphiti-core version. The library
is async-first, and this adapter bridges with asyncio.run(), so use it from
sync code only (not inside a running event loop).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Sequence

from ..types import MemoryRecord, MemoryType, Scope, ScopeRef


class GraphitiStore:
    OWNS_EXTRACTION = True  # Graphiti extracts entities/edges from episodes itself

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        try:
            from graphiti_core import Graphiti
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "GraphitiStore requires 'graphiti-core': pip install graphiti-core"
            ) from exc
        self.graphiti = Graphiti(uri, user, password)
        asyncio.run(self.graphiti.build_indices_and_constraints())

    def add(self, record: MemoryRecord) -> MemoryRecord:
        from graphiti_core.nodes import EpisodeType

        asyncio.run(
            self.graphiti.add_episode(
                name=record.id,
                episode_body=record.text,
                source=EpisodeType.text,
                source_description=record.source or "memory-fabric",
                reference_time=datetime.fromtimestamp(
                    record.valid_from, tz=timezone.utc
                ),
                group_id=f"{record.scope.value}:{record.scope_id}",
            )
        )
        return record

    def get(self, memory_id: str) -> MemoryRecord | None:
        raise NotImplementedError("GraphitiStore does not support get-by-id")

    def delete(self, memory_id: str) -> bool:
        raise NotImplementedError("Delete episodes via the Graphiti API directly")

    def search_keyword(
        self,
        query: str,
        scopes: Sequence[ScopeRef] | None = None,
        types: Sequence[MemoryType] | None = None,
        limit: int = 20,
        include_invalid: bool = False,
    ) -> list[MemoryRecord]:
        # Graphiti's search is already hybrid (semantic + BM25 + graph
        # traversal); expose it on this channel and leave the vector channel
        # empty to avoid double counting in fusion.
        group_ids = (
            [f"{scope.value}:{scope_id}" for scope, scope_id in scopes]
            if scopes
            else None
        )
        edges = asyncio.run(
            self.graphiti.search(query, group_ids=group_ids, num_results=limit)
        )
        records: list[MemoryRecord] = []
        for edge in edges:
            scope, _, scope_id = (edge.group_id or "user::default").partition(":")
            records.append(
                MemoryRecord(
                    id=edge.uuid,
                    text=edge.fact,
                    memory_type=MemoryType.SEMANTIC,
                    scope=Scope(scope) if scope in Scope._value2member_map_ else Scope.USER,
                    scope_id=scope_id or "default",
                    valid_from=(
                        edge.valid_at.timestamp() if edge.valid_at else 0.0
                    ),
                    valid_to=(
                        edge.invalid_at.timestamp() if edge.invalid_at else None
                    ),
                )
            )
        if not include_invalid:
            records = [r for r in records if r.is_valid]
        return records

    def search_recent(self, scopes=None, types=None, limit=20):
        return []  # Graphiti has no cheap "most recent episodes" API; use search

    def search_vector(self, query, scopes=None, types=None, limit=20):
        return []  # folded into search_keyword (Graphiti search is already hybrid)

    def find_active_fact(self, scope, subject, predicate):
        raise NotImplementedError("Graphiti resolves fact conflicts internally")

    def invalidate(self, memory_id, superseded_by=None):
        raise NotImplementedError("Graphiti invalidates edges internally")

    def history(self, subject, predicate, scopes=None):
        raise NotImplementedError(
            "Query Graphiti directly for edge validity intervals"
        )
