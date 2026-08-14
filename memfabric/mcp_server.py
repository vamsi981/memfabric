"""MCP server exposing a MemFabric instance to any MCP client
(Claude Code, Claude Desktop, Cursor, ...).

Install and run:

    pip install memfabric[mcp]
    memfabric-mcp                # stdio transport

Configuration (environment variables):

    MEMFABRIC_DB      SQLite path            (default: ./memfabric.db)
    MEMFABRIC_SCOPE   default "scope:id"     (default: user:default)
    MEMFABRIC_LLM     provider spec, e.g. "ollama:llama3.1"  (optional)

Claude Code registration:

    claude mcp add memfabric -- memfabric-mcp

SECURITY NOTE: scopes here are an organizational primitive, not an access-
control boundary. Any client of this server can pass any scope. Run one
server per trust domain, or enforce identity at a layer above.
"""

from __future__ import annotations

import os

from .fabric import MemoryFabric
from .types import MemoryType, Scope


def _build_fabric() -> MemoryFabric:
    raw = os.environ.get("MEMFABRIC_SCOPE", "user:default")
    scope_name, _, scope_id = raw.partition(":")
    return MemoryFabric(
        db_path=os.environ.get("MEMFABRIC_DB", "memfabric.db"),
        default_scope=(Scope(scope_name), scope_id or "default"),
    )


def _parse_scopes(scopes: str) -> list[tuple[Scope, str]] | None:
    """Parse "user:vamsi,team:platform" into scope refs; empty -> default."""
    if not scopes.strip():
        return None
    refs = []
    for part in scopes.split(","):
        scope_name, _, scope_id = part.strip().partition(":")
        refs.append((Scope(scope_name), scope_id or "default"))
    return refs


def create_server(fabric: MemoryFabric | None = None):
    """Build the FastMCP server (separated from main() so it's testable)."""
    try:
        from mcp.server.mcpserver import MCPServer  # mcp >= 2.0
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as MCPServer  # mcp 1.x
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "The MCP server requires the 'mcp' package: pip install memfabric[mcp]"
            ) from exc

    fabric = fabric or _build_fabric()
    server = MCPServer("memfabric")

    @server.tool()
    def remember(
        text: str,
        memory_type: str = "semantic",
        subject: str = "",
        predicate: str = "",
        object: str = "",
        scopes: str = "",
    ) -> str:
        """Store one memory. Fill subject/predicate/object for a concrete fact
        (e.g. "Project X" / "runs_on" / "Azure Foundry") to enable temporal
        tracking: a new fact supersedes the old one instead of deleting it.
        memory_type: semantic | episodic | procedural.
        scopes: optional single "scope:id" (user/session/agent/team/project/org)."""
        refs = _parse_scopes(scopes)
        scope, scope_id = refs[0] if refs else (None, None)
        record = fabric.remember(
            text,
            memory_type=MemoryType(memory_type),
            scope=scope,
            scope_id=scope_id,
            subject=subject or None,
            predicate=predicate or None,
            object=object or None,
            source="mcp",
        )
        return f"stored {record.id}: {record.describe()}"

    @server.tool()
    def ingest(text: str, role: str = "user", scopes: str = "") -> str:
        """Ingest a conversation turn: stores it as an episode and, when an
        LLM provider is configured, extracts durable facts from it."""
        refs = _parse_scopes(scopes)
        scope, scope_id = refs[0] if refs else (None, None)
        created = fabric.ingest(text, role=role, scope=scope, scope_id=scope_id, source="mcp")
        return "\n".join(r.describe() for r in created) or "nothing stored"

    @server.tool()
    def recall(query: str, scopes: str = "", limit: int = 8) -> str:
        """Search memories (hybrid keyword+recency retrieval, currently-valid
        facts only). scopes: comma-separated allowlist like
        "user:vamsi,team:platform"; empty uses the server's default scope."""
        hits = fabric.recall(query, scopes=_parse_scopes(scopes), limit=limit)
        return "\n".join(f"{s.score:.4f} {s.record.describe()}" for s in hits) or "no matches"

    @server.tool()
    def history(subject: str, predicate: str, scopes: str = "") -> str:
        """Show how a fact changed over time (full supersede chain, oldest
        first), e.g. subject="Project X", predicate="runs_on"."""
        chain = fabric.history(subject, predicate, scopes=_parse_scopes(scopes))
        return "\n".join(r.describe() for r in chain) or "no history for that fact"

    @server.tool()
    def build_context(query: str, scopes: str = "", char_budget: int = 8000) -> str:
        """Assemble a prompt-ready <memory_context> block for a query:
        relevant knowledge + relevant events, within the character budget."""
        return (
            fabric.build_context(
                query,
                scopes=_parse_scopes(scopes),
                char_budget=char_budget,
                include_working=False,
            )
            or "no relevant memories"
        )

    @server.tool()
    def forget(memory_id: str) -> str:
        """Hard-delete a memory by id. For facts that merely changed, prefer
        storing the new fact; supersede keeps the history."""
        return "deleted" if fabric.forget(memory_id) else "not found"

    return server


def main() -> None:
    create_server().run()


if __name__ == "__main__":
    main()
