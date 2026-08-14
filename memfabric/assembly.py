"""Context assembly: turn working memory + recalled memories into one
prompt-ready block, within a character budget.

Sections are budgeted in priority order (working memory > facts > episodes)
so the most load-bearing context survives truncation.
"""

from __future__ import annotations

from .types import MemoryType, ScoredMemory
from .working_memory import WorkingMemory


def assemble_context(
    recalled: list[ScoredMemory],
    working: WorkingMemory | None = None,
    char_budget: int = 8000,
    tag: str = "memory_context",
) -> str:
    sections: list[tuple[str, str]] = []

    if working is not None:
        rendered = working.render()
        if rendered:
            sections.append(("Working memory", rendered))

    facts = [
        s.record.describe()
        for s in recalled
        if s.record.memory_type in (MemoryType.SEMANTIC, MemoryType.PROCEDURAL)
    ]
    if facts:
        sections.append(("Relevant knowledge", "\n".join(f"- {f}" for f in facts)))

    episodes = [
        s.record.describe()
        for s in recalled
        if s.record.memory_type is MemoryType.EPISODIC
    ]
    if episodes:
        sections.append(("Relevant events", "\n".join(f"- {e}" for e in episodes)))

    if not sections:
        return ""

    remaining = char_budget
    parts: list[str] = []
    for title, body in sections:
        block = f"## {title}\n{body}"
        if len(block) > remaining:
            block = block[: max(remaining, 0)]
        if block:
            parts.append(block)
            remaining -= len(block) + 2
        if remaining <= 0:
            break

    return f"<{tag}>\n" + "\n\n".join(parts) + f"\n</{tag}>"
