"""Model-agnostic LLM providers for the Memory Fabric.

Pick a provider three ways:

1. Spec string (in MemoryFabric(llm=...) or the MEMFABRIC_LLM env var):
     "anthropic"                    Claude, default model
     "anthropic:claude-opus-5"     explicit model
     "openai:gpt-5-mini"           OpenAI
     "ollama:llama3.1"             local Ollama, no API key
2. An instance:  MemoryFabric(llm=OpenAICompatibleLLM(base_url=..., model=...))
3. Anything implementing the MemoryLLM protocol (extract + rerank).

Auto-detection ("auto", the default) resolves in order: MEMFABRIC_LLM
spec, ANTHROPIC_API_KEY, OPENAI_API_KEY, else no LLM (heuristic mode).
"""

from __future__ import annotations

import os

from .anthropic_llm import AnthropicLLM
from .base import (
    BaseMemoryLLM,
    ExtractedMemory,
    ExtractionResult,
    LLMUnavailable,
    MemoryLLM,
    RerankResult,
    parse_structured,
    schema_prompt,
)
from .openai_llm import OpenAICompatibleLLM, ollama

__all__ = [
    "AnthropicLLM",
    "BaseMemoryLLM",
    "ExtractedMemory",
    "ExtractionResult",
    "LLMUnavailable",
    "MemoryLLM",
    "OpenAICompatibleLLM",
    "RerankResult",
    "auto_llm",
    "create_llm",
    "ollama",
    "parse_structured",
    "schema_prompt",
]


def create_llm(spec: str, model: str | None = None) -> MemoryLLM:
    """Build a provider from a "provider[:model]" spec string."""
    provider, _, spec_model = spec.partition(":")
    chosen = spec_model or model
    kwargs = {"model": chosen} if chosen else {}
    if provider == "anthropic":
        return AnthropicLLM(**kwargs)
    if provider == "openai":
        return OpenAICompatibleLLM(**kwargs)
    if provider == "ollama":
        return ollama(**kwargs)
    raise ValueError(
        f"unknown LLM provider {provider!r} (expected anthropic | openai | ollama; "
        "for anything else, pass a MemoryLLM instance)"
    )


def auto_llm(model: str | None = None) -> MemoryLLM | None:
    """Env-driven provider detection; None means heuristic mode."""
    spec = os.environ.get("MEMFABRIC_LLM")
    if spec:
        return create_llm(spec, model)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicLLM(**({"model": model} if model else {}))
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAICompatibleLLM(**({"model": model} if model else {}))
    return None
