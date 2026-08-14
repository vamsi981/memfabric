"""MemFabric: a temporal, permission-scoped memory layer for AI agents.
SQLite-backed by default; Mem0 and Graphiti pluggable; any LLM provider
(Anthropic, OpenAI, OpenAI-compatible, or your own) for extraction and
reranking."""

from .fabric import MemoryFabric
from .types import MemoryRecord, MemoryType, Scope, ScoredMemory
from .working_memory import WorkingMemory

__all__ = [
    "MemoryFabric",
    "MemoryRecord",
    "MemoryType",
    "Scope",
    "ScoredMemory",
    "WorkingMemory",
]

__version__ = "0.1.0"
