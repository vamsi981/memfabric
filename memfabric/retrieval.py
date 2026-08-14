"""Hybrid retrieval: run every channel the backend supports, fuse with
Reciprocal Rank Fusion, optionally rerank the head with the configured LLM."""

from __future__ import annotations

from typing import Mapping, Sequence

from .llms import LLMUnavailable, MemoryLLM
from .stores.base import MemoryStore
from .types import MemoryRecord, MemoryType, ScopeRef, ScoredMemory

DEFAULT_CHANNEL_WEIGHTS = {
    "keyword": 1.0,
    "vector": 1.0,
    "recency": 0.5,  # recency is a tiebreaker, not a relevance signal
}


def reciprocal_rank_fusion(
    channels: Mapping[str, Sequence[MemoryRecord]],
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[ScoredMemory]:
    weights = weights or DEFAULT_CHANNEL_WEIGHTS
    fused: dict[str, ScoredMemory] = {}
    for channel, records in channels.items():
        w = weights.get(channel, 1.0)
        for rank, record in enumerate(records):
            entry = fused.get(record.id)
            if entry is None:
                entry = fused[record.id] = ScoredMemory(record=record, score=0.0)
            entry.score += w / (k + rank + 1)
            entry.origins.append(channel)
    return sorted(fused.values(), key=lambda s: s.score, reverse=True)


class HybridRetriever:
    def __init__(
        self,
        store: MemoryStore,
        llm: MemoryLLM | None = None,
        rerank_top: int = 12,
    ):
        self.store = store
        self.llm = llm
        self.rerank_top = rerank_top

    def retrieve(
        self,
        query: str,
        scopes: Sequence[ScopeRef] | None = None,
        types: Sequence[MemoryType] | None = None,
        limit: int = 8,
        rerank: bool = True,
    ) -> list[ScoredMemory]:
        channels: dict[str, list[MemoryRecord]] = {
            "keyword": self.store.search_keyword(query, scopes, types, limit=limit * 3),
            "vector": self.store.search_vector(query, scopes, types, limit=limit * 3),
            "recency": self.store.search_recent(scopes, types, limit=limit),
        }
        fused = reciprocal_rank_fusion(channels)
        if rerank and self.llm is not None and len(fused) > 1:
            fused = self._rerank(query, fused)
        return fused[:limit]

    def _rerank(self, query: str, fused: list[ScoredMemory]) -> list[ScoredMemory]:
        head, tail = fused[: self.rerank_top], fused[self.rerank_top :]
        try:
            order = self.llm.rerank(query, [s.record.text for s in head])
        except LLMUnavailable:
            return fused
        reranked = [head[i] for i in order]
        for item in reranked:
            item.origins.append("rerank")
        dropped = [s for i, s in enumerate(head) if i not in set(order)]
        return reranked + dropped + tail
