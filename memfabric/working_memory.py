"""Letta-style working memory: what the agent should hold in its context
right now: current goal, key facts, recent turns. In-process (per agent
run), not persisted; durable knowledge belongs in the stores."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class Turn:
    role: str
    text: str


class WorkingMemory:
    def __init__(self, char_budget: int = 4000, max_turns: int = 12):
        self.char_budget = char_budget
        self.blocks: dict[str, str] = {}
        self.turns: deque[Turn] = deque(maxlen=max_turns)

    def set_block(self, label: str, content: str) -> None:
        """Create or replace a named block, e.g. 'goal', 'constraints'."""
        self.blocks[label] = content.strip()

    def append_block(self, label: str, content: str) -> None:
        existing = self.blocks.get(label, "")
        self.blocks[label] = (existing + "\n" + content.strip()).strip()

    def remove_block(self, label: str) -> None:
        self.blocks.pop(label, None)

    def add_turn(self, role: str, text: str) -> None:
        self.turns.append(Turn(role=role, text=text))

    def render(self) -> str:
        parts: list[str] = []
        for label, content in self.blocks.items():
            parts.append(f"### {label}\n{content}")
        if self.turns:
            lines = "\n".join(f"- {t.role}: {t.text}" for t in self.turns)
            parts.append(f"### recent turns\n{lines}")
        text = "\n\n".join(parts)
        if len(text) > self.char_budget:
            text = text[-self.char_budget :]
            # avoid starting mid-line after the cut
            newline = text.find("\n")
            if 0 <= newline < 200:
                text = text[newline + 1 :]
        return text
