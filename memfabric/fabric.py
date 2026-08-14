"""MemoryFabric, the unified memory API.

    fabric = MemoryFabric()                          # SQLite, zero infra
    fabric.remember("Vamsi prefers Python", scope_id="vamsi")
    fabric.ingest("We migrated Project X to Azure Foundry", role="user")
    fabric.recall("what does Project X run on")
    fabric.build_context("Project X infrastructure")  # prompt-ready block

Backends are pluggable (LocalStore default; Mem0Store / GraphitiStore
adapters). Extraction and reranking run on any LLM provider (Anthropic,
OpenAI, or anything OpenAI-compatible such as Ollama, Groq, or vLLM),
selected via spec string, env vars, or an injected MemoryLLM instance.
Everything degrades gracefully with no provider at all.
"""

from __future__ import annotations

from typing import Literal, Sequence

from .assembly import assemble_context
from .llms import LLMUnavailable, MemoryLLM, auto_llm, create_llm
from .retrieval import HybridRetriever
from .stores.base import MemoryStore
from .stores.local_store import Embedder, LocalStore
from .types import MemoryRecord, MemoryType, Scope, ScopeRef, ScoredMemory
from .working_memory import WorkingMemory


class MemoryFabric:
    def __init__(
        self,
        store: MemoryStore | None = None,
        llm: MemoryLLM | str | Literal["auto"] | None = "auto",
        db_path: str = "memfabric.db",
        embedder: Embedder | None = None,
        default_scope: ScopeRef = (Scope.USER, "default"),
        model: str | None = None,
    ):
        self.store = store or LocalStore(db_path, embedder=embedder)
        self.default_scope = default_scope
        self.working = WorkingMemory()
        self._store_owns_extraction = getattr(self.store, "OWNS_EXTRACTION", False)

        if llm == "auto":
            # Env-driven detection; heuristic mode when nothing resolves.
            try:
                self.llm = auto_llm(model)
            except (LLMUnavailable, ValueError):
                self.llm = None
        elif isinstance(llm, str):
            # Explicit spec ("openai:gpt-5-mini", "ollama:llama3.1", ...):
            # let errors surface, the caller asked for this provider.
            self.llm = create_llm(llm, model)
        else:
            self.llm = llm

        self.retriever = HybridRetriever(self.store, llm=self.llm)

    # -- write path ---------------------------------------------------------

    def remember(
        self,
        text: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        scope: Scope | None = None,
        scope_id: str | None = None,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        source: str | None = None,
        metadata: dict | None = None,
    ) -> MemoryRecord:
        """Store one memory. Facts with a (subject, predicate) key supersede
        the previously valid fact instead of piling up: the old record keeps
        its history (valid_from → valid_to, superseded_by) and drops out of
        default recall."""
        scope_ref = self._resolve_scope(scope, scope_id)
        record = MemoryRecord(
            text=text,
            memory_type=memory_type,
            scope=scope_ref[0],
            scope_id=scope_ref[1],
            subject=subject,
            predicate=predicate,
            object=object,
            source=source,
            metadata=metadata or {},
        )
        if subject and predicate:
            try:
                active = self.store.find_active_fact(scope_ref, subject, predicate)
            except NotImplementedError:
                active = None
            if active is not None:
                if _same_object(active.object, object):
                    return active  # already known; no-op
                self.store.invalidate(active.id, superseded_by=record.id)
        return self.store.add(record)

    def ingest(
        self,
        text: str,
        role: str = "user",
        scope: Scope | None = None,
        scope_id: str | None = None,
        source: str | None = None,
        keep_episode: bool = True,
    ) -> list[MemoryRecord]:
        """Ingest a conversation turn: record it as an episode and, when an
        LLM provider is available, extract durable semantic/procedural
        memories from it (with temporal supersede on repeated facts)."""
        scope_ref = self._resolve_scope(scope, scope_id)
        created: list[MemoryRecord] = []

        self.working.add_turn(role, text)

        if keep_episode and not self._store_owns_extraction:
            created.append(
                self.remember(
                    f"{role} said: {text}",
                    memory_type=MemoryType.EPISODIC,
                    scope=scope_ref[0],
                    scope_id=scope_ref[1],
                    source=source,
                )
            )

        if self._store_owns_extraction:
            # Backends like Mem0/Graphiti run their own extraction pipeline.
            created.append(
                self.remember(
                    text,
                    scope=scope_ref[0],
                    scope_id=scope_ref[1],
                    source=source,
                )
            )
            return created

        if self.llm is not None:
            try:
                extracted = self.llm.extract(text, role=role)
            except LLMUnavailable:
                self.llm = None  # stop retrying a dead client this session
                self.retriever.llm = None
                extracted = []
            for item in extracted:
                created.append(
                    self.remember(
                        item.text,
                        memory_type=MemoryType(item.memory_type),
                        scope=scope_ref[0],
                        scope_id=scope_ref[1],
                        subject=item.subject,
                        predicate=item.predicate,
                        object=item.object,
                        source=source,
                    )
                )
        return created

    def forget(self, memory_id: str) -> bool:
        return self.store.delete(memory_id)

    # -- read path ----------------------------------------------------------

    def recall(
        self,
        query: str,
        scopes: Sequence[ScopeRef] | None = None,
        types: Sequence[MemoryType] | None = None,
        limit: int = 8,
        rerank: bool = True,
    ) -> list[ScoredMemory]:
        """Permission-aware hybrid recall. `scopes` is the caller's read
        allowlist; it defaults to the fabric's default scope."""
        scopes = scopes or [self.default_scope]
        return self.retriever.retrieve(query, scopes, types, limit, rerank=rerank)

    def history(
        self,
        subject: str,
        predicate: str,
        scopes: Sequence[ScopeRef] | None = None,
    ) -> list[MemoryRecord]:
        """How a fact changed over time, oldest first."""
        return self.store.history(subject, predicate, scopes or [self.default_scope])

    def build_context(
        self,
        query: str,
        scopes: Sequence[ScopeRef] | None = None,
        limit: int = 8,
        char_budget: int = 8000,
        include_working: bool = True,
    ) -> str:
        """One prompt-ready block: working memory + relevant knowledge +
        relevant events, inside <memory_context> tags."""
        recalled = self.recall(query, scopes=scopes, limit=limit)
        return assemble_context(
            recalled,
            working=self.working if include_working else None,
            char_budget=char_budget,
        )

    # -- internals ----------------------------------------------------------

    def _resolve_scope(self, scope: Scope | None, scope_id: str | None) -> ScopeRef:
        return (
            scope if scope is not None else self.default_scope[0],
            scope_id if scope_id is not None else self.default_scope[1],
        )


def _same_object(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    return a.strip().lower() == b.strip().lower()
