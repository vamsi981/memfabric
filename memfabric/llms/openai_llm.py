"""OpenAI-compatible provider. Requires `pip install -e .[openai]`.

One class covers every endpoint that speaks the OpenAI chat-completions
protocol: OpenAI itself, Ollama, vLLM, LM Studio, Groq, OpenRouter, Together,
Azure OpenAI (via base_url), and most self-hosted gateways.

Structured output is prompt-based (schema appended to the prompt, reply
parsed tolerantly) so it works uniformly across all of these, including
small local models with no response_format support.
"""

from __future__ import annotations

from pydantic import BaseModel

from .base import (
    BaseMemoryLLM,
    LLMUnavailable,
    is_permanent_error,
    parse_structured,
    schema_prompt,
)

DEFAULT_MODEL = "gpt-5-mini"


class OpenAICompatibleLLM(BaseMemoryLLM):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        api_key: str | None = None,
        client=None,
    ):
        if client is None:
            try:
                import openai
            except ImportError as exc:
                raise LLMUnavailable("openai package not installed") from exc
            try:
                client = openai.OpenAI(
                    **({"base_url": base_url} if base_url else {}),
                    **({"api_key": api_key} if api_key else {}),
                )
            except Exception as exc:
                raise LLMUnavailable(f"could not build OpenAI client: {exc}") from exc
        self.client = client
        self.model = model

    def _structured(self, prompt: str, output_format: type[BaseModel]):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": schema_prompt(prompt, output_format)}
                ],
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:
            if is_permanent_error(exc):
                raise LLMUnavailable(f"OpenAI-compatible call failed: {exc}") from exc
            return None  # transient (rate limit, overload, network); skip this call
        return parse_structured(content, output_format)


def ollama(model: str = "llama3.1", host: str = "http://localhost:11434") -> OpenAICompatibleLLM:
    """Local Ollama via its OpenAI-compatible endpoint. No API key needed."""
    return OpenAICompatibleLLM(model=model, base_url=f"{host}/v1", api_key="ollama")
