"""Storage backend protocol.

LocalStore implements the whole protocol (including the temporal-fact
operations). Adapter backends (Mem0, Graphiti) implement what their engine
supports and raise NotImplementedError for the rest; MemoryFabric degrades
gracefully around those gaps.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..types import MemoryRecord, MemoryType, ScopeRef


@runtime_checkable
class MemoryStore(Protocol):
    def add(self, record: MemoryRecord) -> MemoryRecord:
        """Persist a record and return it."""
        ...

    def get(self, memory_id: str) -> MemoryRecord | None: ...

    def delete(self, memory_id: str) -> bool:
        """Hard-delete (forget). Prefer invalidate() for facts that changed."""
        ...

    def search_keyword(
        self,
        query: str,
        scopes: Sequence[ScopeRef] | None = None,
        types: Sequence[MemoryType] | None = None,
        limit: int = 20,
        include_invalid: bool = False,
    ) -> list[MemoryRecord]: ...

    def search_recent(
        self,
        scopes: Sequence[ScopeRef] | None = None,
        types: Sequence[MemoryType] | None = None,
        limit: int = 20,
    ) -> list[MemoryRecord]: ...

    def search_vector(
        self,
        query: str,
        scopes: Sequence[ScopeRef] | None = None,
        types: Sequence[MemoryType] | None = None,
        limit: int = 20,
    ) -> list[MemoryRecord]:
        """Semantic similarity search. Backends without an embedder return []."""
        ...

    # -- temporal fact operations -------------------------------------------

    def find_active_fact(
        self, scope: ScopeRef, subject: str, predicate: str
    ) -> MemoryRecord | None:
        """The currently-valid fact for (subject, predicate) in a scope."""
        ...

    def invalidate(
        self, memory_id: str, superseded_by: str | None = None
    ) -> MemoryRecord | None:
        """Close a fact's validity window instead of deleting it."""
        ...

    def history(
        self, subject: str, predicate: str, scopes: Sequence[ScopeRef] | None = None
    ) -> list[MemoryRecord]:
        """Full temporal chain for (subject, predicate), oldest first."""
        ...
