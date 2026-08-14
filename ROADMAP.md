# Roadmap

Concrete plan for what gets built, in order. Each milestone lists its scope
and the check for "done". Versions follow the release flow: green CI, tag,
automatic PyPI publish.

## v0.3.0: reflect() consolidation

The answer to "how does it forget?", which is the question every memory
system gets asked first.

Scope:

- `fabric.reflect(scopes)` runs three passes, all working without an LLM:
  1. Dedup: semantic facts with identical `(subject_key, predicate_key)`
     and a normalized-equal object collapse to the oldest record; plain
     text memories with normalized-equal text collapse the same way.
  2. Promotion: when the same fact-shaped observation appears in N or more
     episodes (default 3), promote it to a semantic fact with the episodes
     recorded as provenance. With an LLM configured, promotion can also
     summarize an episode cluster into one fact.
  3. Pruning: `prune_episodes(older_than=days, keep=n)` deletes old
     episodes past a retention window, never touching superseded facts
     (history is the product, not the waste).
- A `reflect` tool on the MCP server.
- Report object returned: counts of merged, promoted, pruned.

Done when: reflect() is deterministic without an LLM, covered by offline
tests, and a fabric that ingested 100 duplicate-ish turns shrinks to a
bounded memory count while recall quality holds in tests.

## v0.4.0: vector channel out of the box

Today the vector channel needs a hand-rolled embedder callable. Make it a
config value.

Scope:

- `memfabric[embeddings]` extra installing sentence-transformers; embedder
  specs like `st:all-MiniLM-L6-v2` and `voyage:voyage-3`, plus a
  `MEMFABRIC_EMBEDDER` env var, mirroring how LLM providers resolve.
- Embedding writes move off the hot path: cache table keyed by text hash,
  batched embedding calls, `fabric.reindex()` to backfill existing rows.
- When the `sqlite-vec` extension is present, use its ANN index instead of
  brute-force cosine; fall back silently when absent.

Done when: `MemoryFabric(embedder="st:all-MiniLM-L6-v2")` works with no
other code, reindex backfills a v0.3 database, and the vector channel
participates in fusion in tests (with a fake embedder, offline).

## v0.5.0: lifecycle and single-node scale

Push the SQLite ceiling before adding servers.

Scope:

- WAL journal mode and busy_timeout defaults for concurrent readers.
- `last_accessed` / `access_count` tracking on recall, with an optional
  usage-aware boost in the recency channel.
- Per-scope TTL policies (e.g. session memories expire in days, org
  memories never), enforced by reflect().
- Scope hierarchies: recall with `(ORG, "acme")` optionally includes team
  and user scopes beneath it via an explicit inheritance map.

Done when: two processes can read one database while one writes, TTLs
expire in tests, and hierarchy recall is covered by tests.

## v0.6.0: the service layer

This is the milestone that closes the "scopes are not security" gap and
turns the honor-system allowlist into enforced isolation.

Scope:

- `memfabric-serve`: a FastAPI wrapper with API-key auth where each key
  maps to a scope allowlist server-side; callers cannot widen their own
  access. Per-tenant database routing (one SQLite file per tenant).
- The MCP server gains the same optional enforcement mode.
- Async fabric API underneath (`AsyncMemoryFabric`), since a server should
  not block its event loop on SQLite.

Done when: an integration test proves a key scoped to `user:alice` cannot
read `user:bob` no matter what it sends, and the README security section
can point at a supported deployment instead of a disclaimer.

## v0.7.0: PostgreSQL backend

The "much larger" move: multi-process, multi-node, managed hosting.

Scope:

- `PostgresStore` implementing the full MemoryStore protocol including
  temporal ops: tsvector for keyword, pgvector for vectors, the same
  key-based supersede semantics.
- `memfabric migrate <sqlite> <postgres-url>` data migration tool.
- CI gains a Postgres service container job.

Done when: the entire offline test suite passes against Postgres in CI via
a store fixture swap, and migration round-trips a populated v0.5 database.

## v0.8.0: graph channel

- Graphiti becomes an additional fusion channel rather than a replacement
  store: relationship-hop results merge into RRF alongside keyword, vector,
  and recency.
- Requires a running Graphiti/Neo4j; always optional.

## Not version-bound (start anytime)

- Launch: Show HN and r/LocalLLaMA, mechanism-first, no benchmark claims.
  Best done after v0.3.0 so the forgetting question has an answer.
- CONTRIBUTING.md and issue templates once outside interest exists.
- A worked agent integration example (Claude Agent SDK or LangGraph).
- LLM-assisted alias resolution ("PG main" is "the postgres db") layered on
  the v0.2 key normalization.

## Non-goals

- Public benchmark claims (LoCoMo-style numbers invite more heat than
  light; we test behavior, not leaderboards).
- More LLM provider classes (the OpenAI-compatible provider already covers
  effectively every endpoint).
- Millions-of-memories scale on a single SQLite file (that's what the
  Postgres backend is for).
