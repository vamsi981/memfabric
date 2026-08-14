"""Tests for canonical fact keys: normalization, fuzzy matching, chain
adoption, and migration of pre-0.2 databases."""

import os
import sqlite3
import tempfile
import unittest

from memfabric import MemoryFabric
from memfabric.normalize import find_similar_key, normalize_key
from memfabric.stores.local_store import LocalStore


def make_fabric(**store_kwargs) -> MemoryFabric:
    return MemoryFabric(store=LocalStore(":memory:", **store_kwargs), llm=None)


class TestNormalizeKey(unittest.TestCase):
    def test_case_and_punctuation(self):
        for variant in ("Project X", "project x", "PROJECT-X", "project_x"):
            self.assertEqual(normalize_key(variant), "project x")

    def test_camel_case_split(self):
        self.assertEqual(normalize_key("ProjectX"), "project x")
        self.assertEqual(normalize_key("AzureFoundry"), "azure foundry")

    def test_predicate_styles(self):
        for variant in ("runs_on", "runs-on", "RunsOn", "runs on"):
            self.assertEqual(normalize_key(variant), "runs on")


class TestFindSimilarKey(unittest.TestCase):
    def test_typo_matches(self):
        self.assertEqual(
            find_similar_key("projct x", ["project x", "team alpha"]), "project x"
        )

    def test_distinct_subjects_do_not_match(self):
        self.assertIsNone(find_similar_key("project y", ["project x"]))

    def test_short_keys_never_fuzzy(self):
        self.assertIsNone(find_similar_key("tam a", ["team a"]))


class TestSupersedeAcrossSpellings(unittest.TestCase):
    def test_camel_case_supersedes(self):
        fabric = make_fabric()
        old = fabric.remember("Project X runs on Azure AI",
                              subject="Project X", predicate="runs_on", object="Azure AI")
        new = fabric.remember("ProjectX moved to Azure Foundry",
                              subject="ProjectX", predicate="RunsOn", object="Azure Foundry")
        self.assertEqual(fabric.store.get(old.id).superseded_by, new.id)

    def test_typo_supersedes_via_fuzzy(self):
        fabric = make_fabric()
        old = fabric.remember("Project X runs on Azure AI",
                              subject="Project X", predicate="runs_on", object="Azure AI")
        new = fabric.remember("Projct X moved to Azure Foundry",
                              subject="Projct X", predicate="runs_on", object="Azure Foundry")
        self.assertEqual(fabric.store.get(old.id).superseded_by, new.id)

    def test_fuzzy_can_be_disabled(self):
        fabric = make_fabric(fuzzy_subjects=False)
        old = fabric.remember("Project X runs on Azure AI",
                              subject="Project X", predicate="runs_on", object="Azure AI")
        fabric.remember("Projct X moved to Azure Foundry",
                        subject="Projct X", predicate="runs_on", object="Azure Foundry")
        self.assertTrue(fabric.store.get(old.id).is_valid)  # two separate chains

    def test_different_subjects_stay_distinct(self):
        fabric = make_fabric()
        x = fabric.remember("Project X runs on Azure",
                            subject="Project X", predicate="runs_on", object="Azure")
        y = fabric.remember("Project Y runs on AWS",
                            subject="Project Y", predicate="runs_on", object="AWS")
        self.assertTrue(fabric.store.get(x.id).is_valid)
        self.assertTrue(fabric.store.get(y.id).is_valid)

    def test_duplicate_object_across_spellings_is_noop(self):
        fabric = make_fabric()
        first = fabric.remember("Project X runs on Azure AI",
                                subject="Project X", predicate="runs_on", object="Azure AI")
        second = fabric.remember("ProjectX runs on Azure-AI",
                                 subject="ProjectX", predicate="runs_on", object="Azure-AI")
        self.assertEqual(first.id, second.id)

    def test_history_is_one_chain_across_spellings(self):
        fabric = make_fabric()
        fabric.remember("v1", subject="Project X", predicate="runs_on", object="A")
        fabric.remember("v2", subject="ProjectX", predicate="RunsOn", object="B")
        fabric.remember("v3", subject="Projct X", predicate="runs_on", object="C")
        chain = fabric.history("PROJECT_X", "runs-on")
        self.assertEqual([r.object for r in chain], ["A", "B", "C"])


_V01_SCHEMA = """
CREATE TABLE memories (
    id            TEXT PRIMARY KEY,
    text          TEXT NOT NULL,
    memory_type   TEXT NOT NULL,
    scope         TEXT NOT NULL,
    scope_id      TEXT NOT NULL,
    subject       TEXT,
    predicate     TEXT,
    object        TEXT,
    created_at    REAL NOT NULL,
    valid_from    REAL NOT NULL,
    valid_to      REAL,
    superseded_by TEXT,
    source        TEXT,
    metadata      TEXT NOT NULL DEFAULT '{}'
);
"""


class TestMigration(unittest.TestCase):
    def test_pre_02_database_is_upgraded_and_backfilled(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            conn = sqlite3.connect(path)
            conn.executescript(_V01_SCHEMA)
            conn.execute(
                "INSERT INTO memories (id, text, memory_type, scope, scope_id,"
                " subject, predicate, object, created_at, valid_from)"
                " VALUES ('old1', 'Project X runs on Azure AI', 'semantic', 'user',"
                " 'default', 'Project X', 'runs_on', 'Azure AI', 1.0, 1.0)"
            )
            conn.commit()
            conn.close()

            store = LocalStore(path)
            record = store.get("old1")
            self.assertEqual(record.subject_key, "project x")
            self.assertEqual(record.predicate_key, "runs on")

            # supersede works against the migrated fact under a new spelling
            fabric = MemoryFabric(store=store, llm=None)
            new = fabric.remember("ProjectX moved on", subject="ProjectX",
                                  predicate="RunsOn", object="Azure Foundry")
            self.assertEqual(store.get("old1").superseded_by, new.id)
            store.close()
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
