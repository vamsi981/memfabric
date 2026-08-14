"""Provider-agnostic LLM layer: fact extraction and recall reranking.

Any object implementing the MemoryLLM protocol (extract + rerank) can power
the fabric. Bring your own provider by subclassing BaseMemoryLLM and
implementing `_structured()`, or by implementing the protocol directly.

All LLM use is optional: MemoryFabric degrades to heuristic behavior when no
provider is available (missing package, missing credentials, API errors), so
the fabric stays usable offline and in tests.

Error semantics:
- `_structured()` returns None for a bad/refused single response (caller
  falls back for that call only)
- `_structured()` raises LLMUnavailable for transport/auth/import failures
  (caller may disable the provider for the session)
"""

from __future__ import annotations

import json
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError


class LLMUnavailable(RuntimeError):
    """Raised when the provider cannot be reached at all; callers fall back."""


class ExtractedMemory(BaseModel):
    text: str
    memory_type: Literal["semantic", "episodic", "procedural"]
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None


class ExtractionResult(BaseModel):
    memories: list[ExtractedMemory]


class RerankResult(BaseModel):
    relevant_indices: list[int]


@runtime_checkable
class MemoryLLM(Protocol):
    """What the fabric needs from a language model. Implement this directly
    to plug in any provider not shipped here."""

    def extract(self, text: str, role: str = "user") -> list[ExtractedMemory]: ...

    def rerank(self, query: str, texts: list[str]) -> list[int]: ...


_EXTRACTION_PROMPT = """\
You maintain the long-term memory of an AI platform. Extract durable memories
from the message below.

Rules:
- Extract only information worth remembering beyond this conversation:
  stable facts, preferences, decisions, relationships, process knowledge.
- Skip greetings, chit-chat, and transient context.
- For a concrete fact, fill subject / predicate / object so the memory system
  can track how the fact changes over time (e.g. subject="Project X",
  predicate="runs_on", object="Azure Foundry"). Use short snake_case
  predicates and reuse the same predicate for the same kind of relation.
- memory_type: "semantic" for facts, "procedural" for how-to/process
  knowledge, "episodic" for notable events.
- Return an empty list when there is nothing worth remembering.

Speaker: {role}
Message:
{text}"""

_RERANK_PROMPT = """\
Rank the memories below by how useful they are for answering the query.
Return relevant_indices: the indices (0-based) of genuinely relevant
memories, most relevant first. Omit irrelevant ones.

Query: {query}

Memories:
{memories}"""


class BaseMemoryLLM:
    """Shared prompting logic; providers implement `_structured()` which
    returns a validated instance of `output_format` (or None on a bad
    response)."""

    def extract(self, text: str, role: str = "user") -> list[ExtractedMemory]:
        result = self._structured(
            _EXTRACTION_PROMPT.format(role=role, text=text), ExtractionResult
        )
        return result.memories if result else []

    def rerank(self, query: str, texts: list[str]) -> list[int]:
        """Return the indices of relevant texts, best first."""
        numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(texts))
        result = self._structured(
            _RERANK_PROMPT.format(query=query, memories=numbered), RerankResult
        )
        if result is None:
            return list(range(len(texts)))
        return [i for i in result.relevant_indices if 0 <= i < len(texts)]

    def _structured(self, prompt: str, output_format: type[BaseModel]):
        raise NotImplementedError


# -- helpers for prompt-based structured output -----------------------------
# For providers without native structured-output APIs, we append the JSON
# schema to the prompt and parse the reply tolerantly.

JSON_INSTRUCTION = """

Respond with ONLY a JSON object matching this schema (no prose, no markdown):
{schema}"""


def schema_prompt(prompt: str, output_format: type[BaseModel]) -> str:
    schema = json.dumps(output_format.model_json_schema())
    return prompt + JSON_INSTRUCTION.format(schema=schema)


def parse_structured(text: str, output_format: type[BaseModel]) -> BaseModel | None:
    """Pull the first JSON object out of a model reply (tolerating code
    fences and surrounding prose) and validate it. None if unparseable."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return output_format.model_validate_json(text[start : end + 1])
    except ValidationError:
        return None
