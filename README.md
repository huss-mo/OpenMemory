<img src="./_assets/icon.png" alt="om logo" width="140"> 

# OpenMemory

**Persistent, semantic memory for AI agents - local-first, framework-agnostic, production-ready.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-224%20passing-brightgreen.svg)](#running-the-test-suite)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![GitHub repo size](https://img.shields.io/github/repo-size/huss-mo/OpenMemory)
![GitHub language count](https://img.shields.io/github/languages/count/huss-mo/OpenMemory)
![GitHub top language](https://img.shields.io/github/languages/top/huss-mo/OpenMemory)

---

## Quick Start

### Option 1 - Docker (recommended)

```bash
git clone https://github.com/huss-mo/OpenMemory && cd OpenMemory
docker compose up -d
# → listening on http://0.0.0.0:4242/mcp
```

### Option 2 - pip

```bash
git clone https://github.com/huss-mo/OpenMemory && cd OpenMemory
pip install -e .
openmemory-mcp
# → listening on http://0.0.0.0:4242/mcp
```

### Connect your client

```json
{
  "mcpServers": {
    "openmemory": {
      "url": "http://<server-ip>:4242/mcp"
    }
  }
}
```

Your agent now has structured, searchable memory that persists across every session - long-term facts, a user profile, agent instructions, an entity graph, and daily logs - all managed automatically. No changes to your agent's code required.

For installation options, embedding providers, multiple workspaces, and the Python API, see [DOCS.md](DOCS.md).

---

## What Becomes Possible

Without memory, every session starts from zero. With OpenMemory, agents can maintain continuity across time, accumulate knowledge, and behave like they actually know the person they're working with.

**A coding assistant that doesn't repeat itself.** It remembers your stack, your preferred patterns, the architectural decisions you've already made, and the approaches you've already ruled out - so it stops re-suggesting the same things.

**A personal assistant that builds a profile over time.** After a few weeks it knows your schedule, your priorities, the people you work with, and how you like to communicate. It doesn't need to ask.

**A research agent that constructs a knowledge graph.** As it reads papers and sources across many sessions, it records entities, relationships, and findings. Searches later return relevant facts regardless of how they were originally worded.

**A customer-facing agent with per-user memory.** Each user gets their own workspace - preferences, history, ongoing context - giving every interaction a personalised, stateful feel without any custom infrastructure.

**A long-running autonomous agent that survives context limits.** When the context window fills, compaction hooks instruct the agent to flush important facts to memory before the window rolls over. The next session picks up exactly where the last one left off.

---

## What OpenMemory Does

Most agents forget everything the moment a conversation ends. They ask the same questions again, repeat the same mistakes, and lose track of the user's preferences and ongoing work. This is not a model limitation - it is missing infrastructure.

OpenMemory provides that infrastructure. It gives your agent a structured, searchable memory that persists across sessions, organised into distinct tiers with clear ownership:

| File | Purpose |
|---|---|
| `MEMORY.md` | Curated long-term facts - preferences, decisions, persistent knowledge. Written by the agent using `memory_write(tier="long_term")`. Survives forever. |
| `USER.md` | Stable user profile - name, role, working style. Edited manually or by the agent. Injected at every session start. |
| `AGENTS.md` | Agent operating instructions - how this agent should behave, what tools to use and when. Seeded with sensible defaults. |
| `RELATIONS.md` | Entity relationship graph - typed triples (`Alice → works_at → Acme Corp`). Written by `memory_relate`, human-readable mirror of the SQLite graph. |
| `daily/YYYY-MM-DD.md` | Append-only daily logs - task progress, running notes, session context. Written by `memory_write(tier="daily")`. |

At session start, all of these files are assembled into a compact system prompt block your agent receives as context - called **bootstrap injection**. At search time, all tiers are queried together or individually.

Additional capabilities:

- **Hybrid search** - BM25 keyword scoring and vector cosine similarity are combined and re-ranked in a single query, so recall is accurate even when the wording differs from what was stored.
- **Zero-setup mode** - with `provider: none`, OpenMemory runs entirely on SQLite with FTS5. No API key, no GPU, no extra dependencies.
- **Pluggable embedding providers** - swap between a local sentence-transformers model, any OpenAI-compatible endpoint (OpenAI, Ollama, LM Studio, LiteLLM), or BM25-only without touching your agent code.
- **Workspace isolation** - each project, user, or agent gets its own directory-backed workspace with independent memory, relations, and daily logs.
- **Relation graph with semantic deduplication** - the graph automatically suppresses near-duplicate triples using configurable cosine similarity thresholding.
- **Compaction hooks** - when a session approaches the context window limit, OpenMemory emits structured prompts that instruct the agent to flush important facts to storage before the window rolls over.

---

## Tools

OpenMemory exposes 9 tools via MCP and the Python API: `memory_bootstrap`, `memory_write`, `memory_search`, `memory_get`, `memory_list`, `memory_delete`, `memory_replace_text`, `memory_replace_lines`, and `memory_relate`.

`MEMORY.md` and all `daily/*.md` files are **append-only** — `memory_delete`, `memory_replace_text`, and `memory_replace_lines` enforce this and will reject edits to those files. Only `USER.md`, `AGENTS.md`, and `RELATIONS.md` are mutable.

**When using the MCP server**, instruct your agent to call `memory_bootstrap` at the start of every session before doing anything else. This loads the full memory context (MEMORY.md, USER.md, AGENTS.md, RELATIONS.md, daily logs) into the conversation. Clients that support the MCP Prompts primitive (Cline, Claude Desktop) can instead use the `memory_bootstrap_prompt` prompt from their Prompts panel.

**When using the Python API**, call `session.bootstrap()` and pass the result as your system prompt — no tool call is needed.

For the full tools reference including parameters, tiers, and source filters, see [DOCS.md - Tools Reference](DOCS.md#tools-reference).

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    AI Agent / LLM                   │
│         (OpenAI, Anthropic, or any framework)       │
└────────────────────┬────────────────────────────────┘
                     │  tool calls + bootstrap prompt
                     ▼
┌─────────────────────────────────────────────────────┐
│                  MemorySession                      │
│   workspace  ·  index  ·  provider  ·  config       │
└───────┬──────────────┬──────────────────────────────┘
        │              │
        ▼              ▼
┌───────────┐  ┌───────────────────────────────────┐
│ Workspace │  │          MemoryIndex              │
│           │  │  SQLite + FTS5 (BM25 keyword)     │
│ MEMORY.md │  │  + optional vector store          │
│ USER.md   │  │  hybrid re-ranking + MMR          │
│ AGENTS.md │  └──────────────┬────────────────────┘
│ RELATIONS │                 │
│ daily/    │                 ▼
└───────────┘  ┌───────────────────────────────────┐
               │       EmbeddingProvider           │
               │  NullProvider  (BM25-only)        │
               │  SentenceTransformer  (local)     │
               │  OpenAICompatible  (HTTP API)     │
               └───────────────────────────────────┘
```

For a detailed breakdown of each layer, the full data flow, and the tech stack, see [DOCS.md - Architecture](DOCS.md#architecture).

---

## Contributing

### Philosophy

OpenMemory is designed around three values:

1. **Simplicity over features.** Every addition must justify its complexity. A zero-dependency BM25-only mode must always work.
2. **Offline-first.** The default configuration must not require an API key, a network connection, or a GPU.
3. **Test-driven.** New behaviour ships with tests. The full suite must pass before any PR is merged.

### Development Setup

```bash
git clone https://github.com/your-org/openmemory.git
cd openmemory

# Install with all dev dependencies
pip install -e ".[dev,local]"

# Or with uv
uv sync --extra dev --extra local
```

### Running the Test Suite

```bash
# Unit tests only (no embedding provider required - fast)
pytest tests/ -m "not embeddings"

# Integration tests (requires a configured embedding provider)
pytest tests/ -m embeddings

# Full suite
pytest tests/

# With coverage
pytest tests/ --cov=openmemory --cov-report=term-missing
```

Integration tests are marked with `@pytest.mark.embeddings` and require an embedding provider to be configured via `openmemory.yaml` or environment variables.

### Submitting a PR

1. Fork the repository and create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes with accompanying tests.
3. Run `pytest tests/ -m "not embeddings"` - all unit tests must pass.
4. Open a pull request with a clear description of what changes and why.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.