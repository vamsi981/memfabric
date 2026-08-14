"""Anthropic (Claude) provider. Requires `pip install -e .[anthropic]`.

Uses the SDK's native structured outputs (messages.parse), so responses are
schema-validated by the API rather than prompt-and-hope."""

from __future__ import annotations

from pydantic import BaseModel

from .base import BaseMemoryLLM, LLMUnavailable, is_permanent_error

DEFAULT_MODEL = "claude-opus-5"


class AnthropicLLM(BaseMemoryLLM):
    def __init__(self, model: str = DEFAULT_MODEL, client=None):
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise LLMUnavailable("anthropic package not installed") from exc
            try:
                client = anthropic.Anthropic()
            except Exception as exc:
                raise LLMUnavailable(f"could not build Anthropic client: {exc}") from exc
        self.client = client
        self.model = model

    def _structured(self, prompt: str, output_format: type[BaseModel]):
        try:
            parse = self.client.messages.parse
        except AttributeError as exc:
            raise LLMUnavailable(
                "installed anthropic SDK has no messages.parse;"
                " upgrade with: pip install -U anthropic"
            ) from exc
        try:
            response = parse(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                output_format=output_format,
            )
        except Exception as exc:
            if is_permanent_error(exc):
                raise LLMUnavailable(f"Anthropic call failed: {exc}") from exc
            return None  # transient (rate limit, overload, network); skip this call
        if response.stop_reason == "refusal":
            return None
        return response.parsed_output
