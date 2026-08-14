"""Adapter-backend unit tests that run without the optional dependencies."""

import asyncio
import unittest

from memfabric.stores.graphiti_store import _run


class TestGraphitiSyncBridge(unittest.TestCase):
    def test_run_outside_an_event_loop(self):
        async def coro():
            return 42

        self.assertEqual(_run(coro()), 42)

    def test_run_inside_a_running_event_loop(self):
        # agent frameworks call the sync store API from async code;
        # asyncio.run() would raise here
        async def inner():
            return 7

        async def outer():
            return _run(inner())

        self.assertEqual(asyncio.run(outer()), 7)


if __name__ == "__main__":
    unittest.main()
