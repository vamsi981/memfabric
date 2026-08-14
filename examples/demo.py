"""End-to-end Memory Fabric demo.

Works with zero configuration (SQLite + keyword retrieval). If an LLM
provider is configured (see .env.example: Anthropic, OpenAI, Ollama, or any
OpenAI-compatible endpoint), it additionally extracts facts from conversation
turns and reranks recall.

Run from the repo root:  python examples/demo.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memfabric import MemoryFabric, Scope


def main() -> None:
    db = os.path.join(tempfile.gettempdir(), "memfabric_demo.db")
    if os.path.exists(db):
        os.remove(db)

    fabric = MemoryFabric(db_path=db, default_scope=(Scope.USER, "vamsi"))
    if fabric.llm:
        provider = type(fabric.llm).__name__
        model = getattr(fabric.llm, "model", "?")
        print(f"LLM provider: {provider} ({model}) - falls back if unreachable\n")
    else:
        print("LLM provider: none detected - heuristic mode (see .env.example)\n")

    # -- 1. Working memory (Letta-style) ------------------------------------
    fabric.working.set_block("goal", "Plan the Project X infrastructure migration")

    # -- 2. Ingest conversation turns (episodic + optional extraction) ------
    print("=== Ingesting conversation ===")
    for role, text in [
        ("user", "We're building Project X for the enterprise platform team."),
        ("user", "Project X currently runs on Azure AI and is owned by Team A."),
        ("assistant", "Noted - I'll track Project X's infrastructure."),
    ]:
        created = fabric.ingest(text, role=role, source="session-001")
        print(f"  [{role}] -> stored {len(created)} memories")

    # -- 3. Explicit temporal facts (works with or without an LLM) ----------
    print("\n=== Fact changes over time ===")
    fabric.remember(
        "Project X runs on Azure AI",
        subject="Project X", predicate="runs_on", object="Azure AI",
    )
    fabric.remember(
        "Project X migrated to Azure Foundry in July 2026",
        subject="Project X", predicate="runs_on", object="Azure Foundry",
    )
    for record in fabric.history("Project X", "runs_on"):
        print(f"  {record.describe()}")

    # -- 4. Scoped memories (enterprise: user vs team vs org) ---------------
    fabric.remember(
        "Deployments must go through the change-review board",
        scope=Scope.ORG, scope_id="acme",
    )
    fabric.remember(
        "Team A deploys via GitHub Actions",
        scope=Scope.TEAM, scope_id="team-a",
    )

    # -- 5. Permission-aware hybrid recall ----------------------------------
    print("\n=== Recall: 'what does Project X run on?' ===")
    for hit in fabric.recall(
        "what does Project X run on?",
        scopes=[(Scope.USER, "vamsi"), (Scope.TEAM, "team-a")],
        limit=5,
    ):
        print(f"  {hit.score:.4f} {hit.origins} {hit.record.text}")

    # -- 6. Assembled context, ready to drop into a prompt ------------------
    print("\n=== Assembled context block ===")
    print(
        fabric.build_context(
            "Project X infrastructure and deployment",
            scopes=[(Scope.USER, "vamsi"), (Scope.TEAM, "team-a"), (Scope.ORG, "acme")],
        )
    )


if __name__ == "__main__":
    main()
