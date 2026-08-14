"""Core data types for the Memory Fabric."""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .normalize import normalize_key


def new_id() -> str:
    return uuid.uuid4().hex


class MemoryType(str, enum.Enum):
    """The kind of memory a record holds.

    SEMANTIC   - durable facts ("Project X runs on Azure Foundry")
    EPISODIC   - things that happened ("user asked about billing on Aug 14")
    PROCEDURAL - how to do things / process preferences ("deploy via GitHub Actions")
    """

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class Scope(str, enum.Enum):
    """Who a memory belongs to. Recall is permission-aware: callers pass the
    scopes they are allowed to read and everything else is invisible."""

    USER = "user"
    SESSION = "session"
    AGENT = "agent"
    TEAM = "team"
    PROJECT = "project"
    ORG = "org"


# A concrete scope is a (Scope, id) pair, e.g. (Scope.USER, "vamsi").
ScopeRef = tuple[Scope, str]


@dataclass
class MemoryRecord:
    """One memory. Semantic facts may carry a (subject, predicate, object)
    triple; when they do, the store applies Graphiti-style temporal
    invalidation: a new fact with the same subject+predicate closes the old
    one (sets valid_to / superseded_by) instead of deleting it."""

    text: str
    memory_type: MemoryType = MemoryType.SEMANTIC
    scope: Scope = Scope.USER
    scope_id: str = "default"
    id: str = field(default_factory=new_id)
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    created_at: float = field(default_factory=time.time)
    valid_from: float = field(default_factory=time.time)
    valid_to: float | None = None  # None => currently valid
    superseded_by: str | None = None  # id of the record that replaced this one
    source: str | None = None  # provenance: session id, doc id, url...
    metadata: dict = field(default_factory=dict)
    # canonical matching keys; auto-derived, but a superseding record adopts
    # its predecessor's keys so a temporal chain never forks on spelling
    subject_key: str | None = None
    predicate_key: str | None = None

    def __post_init__(self) -> None:
        if self.subject and self.subject_key is None:
            self.subject_key = normalize_key(self.subject)
        if self.predicate and self.predicate_key is None:
            self.predicate_key = normalize_key(self.predicate)

    @property
    def is_valid(self) -> bool:
        return self.valid_to is None

    @property
    def triple(self) -> tuple[str, str, str] | None:
        if self.subject and self.predicate and self.object:
            return (self.subject, self.predicate, self.object)
        return None

    def describe(self) -> str:
        """Human/LLM-readable one-liner including temporal validity."""
        since = _fmt_ts(self.valid_from)
        if self.is_valid:
            when = f"since {since}"
        else:
            when = f"was true {since} until {_fmt_ts(self.valid_to)}"
        return f"[{self.memory_type.value}] {self.text} ({when}; scope={self.scope.value}:{self.scope_id})"


@dataclass
class ScoredMemory:
    """A recall result: the record plus its fused relevance score and the
    retrieval channels that surfaced it."""

    record: MemoryRecord
    score: float
    origins: list[str] = field(default_factory=list)


def _fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "?"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
