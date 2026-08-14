# MemFabric

Temporal, permission-scoped memory for AI agents, in a single SQLite file.

Facts change. Most agent memory either overwrites the old fact and loses the
history, or piles up contradictions that confuse retrieval. MemFabric does
what temporal knowledge graphs do, but with zero infrastructure:

```python
from memfabric import MemoryFabric

fabric = MemoryFabric()  # one SQLite file, nothing else

fabric.remember("Project X runs on Azure AI",
                subject="Project X", predicate="runs_on", object="Azure AI")
fabric.remember("Project X moved to Azure Foundry",
                subject="Project X", predicate="runs_on", object="Azure Foundry")

for fact in fabric.history("Project X", "runs_on"):
    print(fact.describe())
# [semantic] Project X runs on Azure AI      (was true 2026-07-01 ... until 2026-08-14; ...)
# [semantic] Project X moved to Azure Foundry (since 2026-08-14; ...)
```

The old fact is superseded, not deleted. It keeps its validity window and a
`superseded_by` pointer, drops out of default recall, and stays queryable as
history. Only currently valid facts reach your prompts.

## What it does

Facts are `(subject, predicate, object)` triples with `valid_from` and
`valid_to` timestamps, the same idea Graphiti uses, except the storage is
stdlib SQLite with FTS5. You don't run Neo4j, you don't run a vector service,
and no LLM is required for any core operation.

Every memory belongs to a `(scope, scope_id)` pair: `user`, `session`,
`agent`, `team`, `project`, or `org`. Recall takes a scope allowlist, and
memories outside it are invisible:

```python
fabric.recall("how do we deploy?",
              scopes=[(Scope.USER, "vamsi"), (Scope.TEAM, "platform")])
```

Retrieval is BM25 plus recency (plus vector search if you plug in an
embedder), fused with Reciprocal Rank Fusion. Because none of that needs a
model, recall is deterministic: you can write unit tests that assert exactly
what your agent remembers. The offline suite runs in under 2 seconds.

When you do configure an LLM, `ingest()` turns conversation turns into
durable facts. Any provider works: Anthropic, OpenAI, an OpenAI-compatible
endpoint (Ollama, Groq, vLLM, OpenRouter, LM Studio), or your own class
implementing a 2-method protocol. Without a provider, turns are stored as
episodes and you add facts through `remember()`.

There is also a Letta-style working memory (named blocks plus a recent-turn
buffer), a context assembler that packs everything into one budgeted
`<memory_context>` block for your prompt, and an MCP server (`memfabric-mcp`)
that exposes the whole thing to Claude Code, Claude Desktop, Cursor, or any
other MCP client.

## What it is not (read this before filing issues)

Scopes are not security. The allowlist is an organizational primitive: your
application decides which scopes a caller may pass, and nothing inside the
library authenticates anyone. If you need enforced multi-tenant isolation,
put MemFabric behind your API boundary (or run one DB per trust domain) and
treat the allowlist as the enforcement point you control. Projects like
Cognee enforce identity server-side; MemFabric deliberately stays a library.

Supersede matches exact `(subject, predicate)` pairs. "Project X" and
"ProjectX" are different subjects today. LLM extraction is prompted to
normalize predicates, but entity resolution is on the roadmap, not in the box.

It is built for thousands to hundreds of thousands of memories, not millions.
Vector search (when you plug in an embedder) is brute-force cosine. For
graph-scale workloads, use the Graphiti adapter and a real graph DB.

There is no automatic forgetting or decay yet. Invalid facts accumulate as
history (that's the point), and episodes accumulate until you prune them.

The Mem0 and Graphiti adapters are experimental: thin mappings onto their
APIs. Verify them against the versions you install.

## Where it sits

| | MemFabric | Graphiti | Mem0 | Cognee |
|---|---|---|---|---|
| Temporal fact supersede | yes | yes (reference impl.) | no | no |
| Permission-scoped recall | allowlist, library-level | namespaces only | namespaces only | server-enforced ACL |
| Zero infrastructure | one SQLite file | needs Neo4j/FalkorDB | needs a vector DB | embedded mode available |
| Works with no LLM at all | yes | no | no | no |
| Scale ceiling | ~10^5 memories | graph-scale | large | large |

If you need graph-scale temporal reasoning, use Graphiti. If you need
server-enforced multi-user ACLs today, use Cognee. If you want temporal facts
plus scoped recall in a library you can `pip install` and unit-test with zero
services running, that's the niche this fills. MemFabric also wraps
[Mem0](https://github.com/mem0ai/mem0) and
[Graphiti](https://github.com/getzep/graphiti) as optional backends behind
the same API, so you can start on SQLite and graduate without rewriting.

## Install

```bash
pip install memfabric                # core: stdlib + pydantic only
pip install memfabric[anthropic]     # + Claude extraction/rerank
pip install memfabric[openai]        # + OpenAI or any OpenAI-compatible endpoint
pip install memfabric[mcp]           # + MCP server
```

## Quickstart

```python
from memfabric import MemoryFabric, Scope

fabric = MemoryFabric(default_scope=(Scope.USER, "vamsi"))

# Facts with temporal tracking
fabric.remember("Deploys go through GitHub Actions",
                subject="deploys", predicate="run_via", object="GitHub Actions",
                scope=Scope.TEAM, scope_id="platform")

# Conversation ingestion (LLM extracts facts when configured)
fabric.ingest("We're migrating Project X to Azure Foundry", role="user")

# Permission-aware hybrid recall
hits = fabric.recall("deployment process",
                     scopes=[(Scope.USER, "vamsi"), (Scope.TEAM, "platform")])

# Prompt-ready context block
block = fabric.build_context("Project X status")
```

Run the full demo (works with zero configuration): `python examples/demo.py`

## LLM providers

```python
fabric = MemoryFabric(llm="ollama:llama3.1")           # local, no API key
fabric = MemoryFabric(llm="anthropic:claude-opus-5")
fabric = MemoryFabric(llm="openai:gpt-5-mini")

# Any OpenAI-compatible endpoint
from memfabric.llms import OpenAICompatibleLLM
fabric = MemoryFabric(llm=OpenAICompatibleLLM(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1", api_key="gsk_..."))

# Bring your own: two methods, no subclassing
class MyLLM:
    def extract(self, text, role="user"): ...
    def rerank(self, query, texts): ...
fabric = MemoryFabric(llm=MyLLM())
```

The default (`llm="auto"`) resolves from the environment: it checks the
`MEMFABRIC_LLM` spec first, then `ANTHROPIC_API_KEY`, then `OPENAI_API_KEY`,
and otherwise runs with no LLM. Anthropic uses native structured outputs.
The OpenAI-compatible provider uses prompt-based JSON with tolerant parsing,
so it behaves the same on hosted APIs and small local models. Verified
against local Ollama (`gpt-oss`).

## MCP server

```bash
pip install memfabric[mcp]
claude mcp add memfabric -- memfabric-mcp
```

Environment: `MEMFABRIC_DB` (SQLite path), `MEMFABRIC_SCOPE` (default
`user:default`), `MEMFABRIC_LLM` (optional provider spec). Tools exposed:
`remember`, `ingest`, `recall`, `history`, `build_context`, `forget`.

## Layout

```
memfabric/
├── fabric.py           MemoryFabric facade (remember/ingest/recall/history/context)
├── types.py            MemoryRecord, MemoryType, Scope, ScoredMemory
├── retrieval.py        hybrid channels + Reciprocal Rank Fusion + rerank hook
├── working_memory.py   Letta-style blocks + recent-turn buffer
├── assembly.py         budgeted <memory_context> builder
├── mcp_server.py       MCP server (memfabric-mcp)
├── llms/               model-agnostic LLM layer (optional)
└── stores/             LocalStore (SQLite) + Mem0/Graphiti adapters
```

## Tests

```bash
python -m unittest discover tests -v    # offline, no LLM, <2s
```

## Roadmap

- Entity/predicate normalization so supersede survives naming drift
- Consolidation pass ("reflect"): merge duplicates, promote episodic facts to semantic
- Out-of-the-box embedders (Voyage, sentence-transformers) for the vector channel
- Scope hierarchies (org inherits to team, team to user)
- Async API and a thin auth-enforcing server wrapper

## Development

Built with AI assistance and reviewed by a human. The design deliberately
borrows from [Graphiti](https://github.com/getzep/graphiti) (temporal
invalidation), [Mem0](https://github.com/mem0ai/mem0) (memory API shape), and
[Letta](https://github.com/letta-ai/letta) (working-memory blocks). The
contribution is the combination, not the parts. Bug reports with failing
tests are the most useful thing you can send.

## License

Apache-2.0
