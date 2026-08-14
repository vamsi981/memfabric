"""Canonical keys for fact matching.

Supersede works by matching (subject, predicate) pairs, so "Project X",
"ProjectX", and "project-x" must resolve to the same temporal chain.
normalize_key() folds case, splits camelCase, and strips punctuation;
find_similar_key() adds a conservative fuzzy layer for typos.
"""

from __future__ import annotations

import difflib
import re

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORDS = re.compile(r"[a-z0-9]+")


def normalize_key(text: str) -> str:
    """Canonical matching key: "ProjectX" -> "project x", "runs_on" ->
    "runs on", "Azure-AI" -> "azure ai"."""
    spaced = _CAMEL_BOUNDARY.sub(" ", text)
    return " ".join(_WORDS.findall(spaced.casefold()))


def find_similar_key(
    key: str,
    candidates: list[str],
    cutoff: float = 0.9,
    min_len: int = 6,
) -> str | None:
    """The closest existing key within cutoff, or None. Short keys are
    excluded entirely: at 5 characters or fewer a single-character edit
    can clear a 0.9 ratio, so "team a"-style keys must match exactly."""
    if len(key) < min_len:
        return None
    pool = [c for c in candidates if len(c) >= min_len and c != key]
    matches = difflib.get_close_matches(key, pool, n=1, cutoff=cutoff)
    return matches[0] if matches else None
