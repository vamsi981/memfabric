"""Tests for the model-agnostic LLM layer (no network calls)."""

import os
import unittest
from unittest import mock

from memfabric import MemoryFabric, MemoryType
from memfabric.llms import (
    ExtractedMemory,
    ExtractionResult,
    LLMUnavailable,
    MemoryLLM,
    RerankResult,
    auto_llm,
    create_llm,
    parse_structured,
    schema_prompt,
)
from memfabric.llms.base import BaseMemoryLLM
from memfabric.stores.local_store import LocalStore


class FakeLLM:
    """A plain object satisfying the MemoryLLM protocol. Shows that a
    provider works without subclassing anything."""

    def extract(self, text, role="user"):
        return [
            ExtractedMemory(
                text="Project X runs on Azure Foundry",
                memory_type="semantic",
                subject="Project X",
                predicate="runs_on",
                object="Azure Foundry",
            )
        ]

    def rerank(self, query, texts):
        return list(range(len(texts)))


class TestProtocol(unittest.TestCase):
    def test_fake_llm_satisfies_protocol(self):
        self.assertIsInstance(FakeLLM(), MemoryLLM)

    def test_fabric_uses_injected_llm(self):
        fabric = MemoryFabric(store=LocalStore(":memory:"), llm=FakeLLM())
        created = fabric.ingest("we moved Project X to Azure Foundry")
        types = sorted(r.memory_type.value for r in created)
        self.assertEqual(types, ["episodic", "semantic"])
        fact = next(r for r in created if r.memory_type is MemoryType.SEMANTIC)
        self.assertEqual(fact.triple, ("Project X", "runs_on", "Azure Foundry"))


class TestPromptBasedParsing(unittest.TestCase):
    def test_parse_clean_json(self):
        result = parse_structured('{"relevant_indices": [2, 0]}', RerankResult)
        self.assertEqual(result.relevant_indices, [2, 0])

    def test_parse_json_with_fences_and_prose(self):
        messy = 'Sure! Here is the JSON:\n```json\n{"memories": []}\n```\nHope that helps.'
        result = parse_structured(messy, ExtractionResult)
        self.assertEqual(result.memories, [])

    def test_parse_garbage_returns_none(self):
        self.assertIsNone(parse_structured("no json here", RerankResult))
        self.assertIsNone(parse_structured('{"wrong_field": 1}', RerankResult))

    def test_schema_prompt_embeds_schema(self):
        prompt = schema_prompt("Rank these.", RerankResult)
        self.assertIn("relevant_indices", prompt)

    def test_base_llm_extract_via_structured_stub(self):
        class StubLLM(BaseMemoryLLM):
            def _structured(self, prompt, output_format):
                return parse_structured(
                    '{"memories": [{"text": "t", "memory_type": "semantic"}]}',
                    output_format,
                )

        memories = StubLLM().extract("anything")
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].memory_type, "semantic")

    def test_rerank_falls_back_on_bad_response(self):
        class NoneLLM(BaseMemoryLLM):
            def _structured(self, prompt, output_format):
                return None

        self.assertEqual(NoneLLM().rerank("q", ["a", "b"]), [0, 1])

    def test_rerank_drops_duplicate_and_out_of_range_indices(self):
        class DupLLM(BaseMemoryLLM):
            def _structured(self, prompt, output_format):
                return RerankResult(relevant_indices=[1, 1, 0, 5, 0])

        self.assertEqual(DupLLM().rerank("q", ["a", "b"]), [1, 0])


class FakeStatusError(Exception):
    """Stands in for SDK errors, which carry the HTTP status_code."""

    def __init__(self, status_code):
        super().__init__(f"http {status_code}")
        self.status_code = status_code


class TestProviderErrorHandling(unittest.TestCase):
    def _anthropic_llm(self, exc):
        from memfabric.llms.anthropic_llm import AnthropicLLM

        class Messages:
            def parse(self, **kwargs):
                raise exc

        class Client:
            messages = Messages()

        return AnthropicLLM(client=Client())

    def test_transient_error_skips_one_call_only(self):
        llm = self._anthropic_llm(FakeStatusError(529))
        self.assertEqual(llm.extract("hi"), [])  # falls back, no raise

    def test_auth_error_raises_unavailable(self):
        llm = self._anthropic_llm(FakeStatusError(401))
        with self.assertRaises(LLMUnavailable):
            llm.extract("hi")

    def test_sdk_without_parse_raises_unavailable(self):
        from memfabric.llms.anthropic_llm import AnthropicLLM

        class OldMessages:
            pass  # no parse method

        class OldClient:
            messages = OldMessages()

        with self.assertRaises(LLMUnavailable):
            AnthropicLLM(client=OldClient()).extract("hi")

    def test_openai_transient_error_skips_one_call_only(self):
        from memfabric.llms.openai_llm import OpenAICompatibleLLM

        class Completions:
            def create(self, **kwargs):
                raise FakeStatusError(429)

        class Chat:
            completions = Completions()

        class Client:
            chat = Chat()

        self.assertEqual(OpenAICompatibleLLM(client=Client()).extract("hi"), [])


class TestProviderResolution(unittest.TestCase):
    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            create_llm("gemini:whatever")

    def test_anthropic_spec(self):
        try:
            llm = create_llm("anthropic:my-model")
        except LLMUnavailable:
            self.skipTest("anthropic package not installed")
        self.assertEqual(llm.model, "my-model")

    def test_openai_and_ollama_specs(self):
        try:
            llm = create_llm("ollama:llama3.1")
        except LLMUnavailable:
            self.skipTest("openai package not installed")
        self.assertEqual(llm.model, "llama3.1")

    def test_auto_returns_none_without_env(self):
        clean = {
            k: v
            for k, v in os.environ.items()
            if k not in ("MEMFABRIC_LLM", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
        }
        with mock.patch.dict(os.environ, clean, clear=True):
            self.assertIsNone(auto_llm())

    def test_fabric_accepts_spec_string(self):
        try:
            fabric = MemoryFabric(
                store=LocalStore(":memory:"), llm="anthropic:my-model"
            )
        except LLMUnavailable:
            self.skipTest("anthropic package not installed")
        self.assertEqual(fabric.llm.model, "my-model")


if __name__ == "__main__":
    unittest.main()
