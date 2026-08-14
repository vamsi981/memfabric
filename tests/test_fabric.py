"""Deterministic tests for the MemFabric core (no LLM, in-memory SQLite).

Run from the repo root:  python -m unittest discover tests -v
"""

import unittest

from memfabric import MemoryFabric, MemoryType, Scope
from memfabric.retrieval import reciprocal_rank_fusion
from memfabric.stores.local_store import LocalStore
from memfabric.types import MemoryRecord


def make_fabric(**kwargs) -> MemoryFabric:
    return MemoryFabric(store=LocalStore(":memory:"), llm=None, **kwargs)


class TestTemporalFacts(unittest.TestCase):
    def test_new_fact_supersedes_old(self):
        fabric = make_fabric()
        old = fabric.remember(
            "Project X runs on Azure AI",
            subject="Project X", predicate="runs_on", object="Azure AI",
        )
        new = fabric.remember(
            "Project X runs on Azure Foundry",
            subject="Project X", predicate="runs_on", object="Azure Foundry",
        )
        stored_old = fabric.store.get(old.id)
        self.assertFalse(stored_old.is_valid)
        self.assertEqual(stored_old.superseded_by, new.id)
        self.assertTrue(fabric.store.get(new.id).is_valid)

    def test_duplicate_fact_is_noop(self):
        fabric = make_fabric()
        first = fabric.remember(
            "Project X runs on Azure AI",
            subject="Project X", predicate="runs_on", object="Azure AI",
        )
        second = fabric.remember(
            "Project X still runs on Azure AI",
            subject="Project X", predicate="runs_on", object="azure ai",
        )
        self.assertEqual(first.id, second.id)
        self.assertEqual(fabric.store.count(), 1)

    def test_history_preserves_chain(self):
        fabric = make_fabric()
        fabric.remember("v1", subject="X", predicate="runs_on", object="A")
        fabric.remember("v2", subject="X", predicate="runs_on", object="B")
        fabric.remember("v3", subject="X", predicate="runs_on", object="C")
        chain = fabric.history("X", "runs_on")
        self.assertEqual([r.object for r in chain], ["A", "B", "C"])
        self.assertEqual([r.is_valid for r in chain], [False, False, True])

    def test_invalid_facts_hidden_from_recall(self):
        fabric = make_fabric()
        fabric.remember("X runs on Azure AI", subject="X", predicate="runs_on", object="Azure AI")
        fabric.remember("X runs on Azure Foundry", subject="X", predicate="runs_on", object="Azure Foundry")
        results = fabric.recall("Azure", rerank=False)
        texts = [s.record.text for s in results]
        self.assertIn("X runs on Azure Foundry", texts)
        self.assertNotIn("X runs on Azure AI", texts)


class TestScopes(unittest.TestCase):
    def test_recall_is_scope_isolated(self):
        fabric = make_fabric()
        fabric.remember("alice likes tennis", scope=Scope.USER, scope_id="alice")
        fabric.remember("bob likes tennis", scope=Scope.USER, scope_id="bob")
        results = fabric.recall("tennis", scopes=[(Scope.USER, "alice")], rerank=False)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].record.scope_id, "alice")

    def test_multi_scope_recall(self):
        fabric = make_fabric()
        fabric.remember("team standard is pytest", scope=Scope.TEAM, scope_id="platform")
        fabric.remember("vamsi prefers unittest", scope=Scope.USER, scope_id="vamsi")
        results = fabric.recall(
            "pytest unittest testing",
            scopes=[(Scope.USER, "vamsi"), (Scope.TEAM, "platform")],
            rerank=False,
        )
        self.assertEqual(len(results), 2)


class TestRecall(unittest.TestCase):
    def test_keyword_recall_finds_relevant(self):
        fabric = make_fabric()
        fabric.remember("deploys go through GitHub Actions", memory_type=MemoryType.PROCEDURAL)
        fabric.remember("the sky is blue")
        results = fabric.recall("how do we deploy", rerank=False)
        self.assertTrue(results)
        self.assertIn("GitHub Actions", results[0].record.text)

    def test_type_filter(self):
        fabric = make_fabric()
        fabric.remember("fact about deploys")
        fabric.remember("user said: deploy it", memory_type=MemoryType.EPISODIC)
        results = fabric.recall("deploy", types=[MemoryType.EPISODIC], rerank=False)
        self.assertTrue(all(s.record.memory_type is MemoryType.EPISODIC for s in results))

    def test_forget(self):
        fabric = make_fabric()
        record = fabric.remember("temporary secret preference")
        self.assertTrue(fabric.forget(record.id))
        self.assertFalse(fabric.recall("secret preference", rerank=False))


class TestFusion(unittest.TestCase):
    def test_rrf_prefers_multi_channel_hits(self):
        a = MemoryRecord(text="in both channels")
        b = MemoryRecord(text="keyword only")
        c = MemoryRecord(text="vector only")
        fused = reciprocal_rank_fusion({"keyword": [b, a], "vector": [c, a]})
        self.assertEqual(fused[0].record.id, a.id)
        self.assertEqual(sorted(fused[0].origins), ["keyword", "vector"])


class TestIngestAndContext(unittest.TestCase):
    def test_ingest_without_llm_keeps_episode(self):
        fabric = make_fabric()
        created = fabric.ingest("we shipped the billing feature", role="user", source="sess-1")
        self.assertEqual(len(created), 1)
        self.assertIs(created[0].memory_type, MemoryType.EPISODIC)
        self.assertEqual(created[0].source, "sess-1")

    def test_build_context_sections(self):
        fabric = make_fabric()
        fabric.working.set_block("goal", "migrate Project X")
        fabric.remember(
            "Project X runs on Azure Foundry",
            subject="Project X", predicate="runs_on", object="Azure Foundry",
        )
        fabric.ingest("user asked about Project X status")
        context = fabric.build_context("Project X")
        self.assertIn("<memory_context>", context)
        self.assertIn("## Working memory", context)
        self.assertIn("migrate Project X", context)
        self.assertIn("Azure Foundry", context)
        self.assertIn("## Relevant events", context)

    def test_context_respects_budget(self):
        fabric = make_fabric()
        for i in range(30):
            fabric.remember(f"fact number {i} about widgets and gadgets")
        context = fabric.build_context("widgets", char_budget=600)
        self.assertLessEqual(len(context), 600 + len("<memory_context>\n\n</memory_context>"))


if __name__ == "__main__":
    unittest.main()
