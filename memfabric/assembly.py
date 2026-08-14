"""Context assembly: turn working memory + recalled memories into one
prompt-ready block, within a character budget.

Sections are budgeted in priority order (working memory > facts > episodes)
so the most load-bearing context survives truncation.
"""

from __future__ import annotations

import re

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

    # stored text is untrusted; a memory containing "</memory_context>"
    # must not be able to close the wrapper and escape the block
    close_tag = re.compile(rf"</(\s*{re.escape(tag)})", re.IGNORECASE)

    remaining = char_budget
    parts: list[str] = []
    for title, body in sections:
        # kept outside the f-string: backslashes in f-string expressions
        # are a SyntaxError before Python 3.12
        safe_body = close_tag.sub(r"<\\/\1", body)
        block = f"## {title}\n{safe_body}"
        if len(block) > remaining:
            # truncate on a line boundary; a half bullet reads as a wrong fact
            cut = block[: max(remaining, 0)]
            newline = cut.rfind("\n")
            block = cut[:newline] if newline > 0 else ""
        if block:
            parts.append(block)
            remaining -= len(block) + 2
        if remaining <= 0:
            break

    return f"<{tag}>\n" + "\n\n".join(parts) + f"\n</{tag}>"
