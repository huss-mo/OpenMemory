<img src="./_assets/icon.png" alt="om logo" width="140"> 

# OpenMemory

**Persistent, semantic memory for AI agents - local-first, framework-agnostic, production-ready.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-108%20passing-brightgreen.svg)](#running-the-test-suite)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![GitHub repo size](https://img.shields.io/github/repo-size/huss-mo/OpenMemory)
![GitHub language count](https://img.shields.io/github/languages/count/huss-mo/OpenMemory)
![GitHub top language](https://img.shields.io/github/languages/top/huss-mo/OpenMemory)

---

## The Problem

As soon as a conversation ends, your agent forgets everything. It asks the same clarifying questions it asked last week. It proposes approaches it already decided against. It loses track of user preferences, ongoing projects, and its own past reasoning. This is not a limitation of the model — it is a missing infrastructure layer.

OpenMemory is that layer.

---

## What OpenMemory Does

OpenMemory gives your agent a structured, searchable memory that persists across sessions. Memory is split into distinct tiers, each with a clear purpose:

| File | Purpose |
|---|---|
| `MEMORY.md` | Curated long-term facts — preferences, decisions, persistent knowledge. Written by the agent using `memory_write(tier="long_term")`. Survives forever. |
| `USER.md` | Stable user profile — name, role, working style. Edited manually or by the agent. Injected at every session start. |
| `AGENTS.md` | Agent operating instructions — how this agent should behave, what tools to use and when. Seeded with sensible defaults. |
| `RELATIONS.md` | Entity relationship graph — typed triples (`Alice → works_at → Acme Corp`). Written by `memory_relate`, human-readable mirror of the SQLite graph. |
| `daily/YYYY-MM-DD.md` | Append-only daily logs — task progress, running notes, session context. Written by `memory_write(tier="daily")`. |

At session start, all of these files are assembled into a compact system prompt block your agent receives as context — called **bootstrap injection**. At search time, all tiers are queried together or individually.

Additional capabilities:

- **Hybrid search** — BM25 keyword scoring and vector cosine similarity are combined and re-ranked in a single query, so recall is accurate even when the wording differs from what was stored.
- **Zero-setup mode** — with `provider: none`, OpenMemory runs entirely on SQLite with FTS5. No API key, no GPU, no extra dependencies.
- **Pluggable embedding providers** — swap between a local sentence-transformers model, any OpenAI-compatible endpoint (OpenAI, Ollama, LM Studio, LiteLLM), or BM25-only without touching your agent code.
- **Workspace isolation** — each project, user, or agent gets its own directory-backed workspace with independent memory, relations, and daily logs.
- **Relation graph with semantic deduplication** — the graph automatically suppresses near-duplicate triples using configurable cosine similarity thresholding.
- **Compaction hooks** — when a session approaches the context window limit, OpenMemory emits structured prompts that instruct the agent to flush important facts to storage before the window rolls over.

---

## Getting Started

### Installation

OpenMemory is not yet published to PyPI. Install directly from source:

```bash
git clone https://github.com/huss-mo/OpenMemory
cd openmemory

# BM25-only (no extra dependencies)
pip install -e .

# With local sentence-transformers embeddings
pip install -e ".[local]"

# Or with uv
uv sync
uv sync --extra local   # for sentence-transformers support
```

### Quickstart

```python
from openmemory.session import MemorySession

session = MemorySession.create("my-project")

# Write a memory
session.execute_tool("memory_write", content="User prefers concise answers.", tier="long_term")

# Search memory
result = session.execute_tool("memory_search", query="communication preferences")
for item in result["data"]["results"]:
    print(item["content"])

# Inject memory context into a system prompt
system_prompt = session.bootstrap()
```

For full integration examples with OpenAI and Anthropic, configuration reference, and environment variables, see [DOCS.md](DOCS.md).

---

## Tools Reference

These tools are registered as JSON schemas for function calling. Use `session.execute_tool(name, **kwargs)` to call them programmatically, or pass `ALL_TOOLS` to your model framework directly.

| Tool | Description | Required Parameters |
|---|---|---|
| `memory_write` | Write a memory to long-term storage (`MEMORY.md`) or today's daily log | `content` |
| `memory_search` | Hybrid semantic + keyword search across all memory tiers | `query` |
| `memory_get` | Retrieve a specific memory chunk by ID | `chunk_id` |
| `memory_list` | List memory chunks with optional source filter and pagination | — |
| `memory_delete` | Delete a specific memory chunk by ID | `chunk_id` |
| `memory_relate` | Record a typed entity relationship (`subject → predicate → object`) | `subject`, `predicate`, `object` |

**`memory_write` tiers:**
- `long_term` — appended to `MEMORY.md`, persists across all sessions
- `daily` — appended to `daily/YYYY-MM-DD.md`, date-stamped journal

**`memory_search` source filters:**
- `long_term`, `daily`, `relations`, `user`, `agents` — restrict to a specific tier, or omit to search all

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    AI Agent / LLM                    │
│         (OpenAI, Anthropic, or any framework)        │
└────────────────────┬────────────────────────────────┘
                     │  tool calls + bootstrap prompt
                     ▼
┌─────────────────────────────────────────────────────┐
│                  MemorySession                       │
│   workspace  ·  index  ·  provider  ·  config        │
└───────┬──────────────┬──────────────────────────────┘
        │              │
        ▼              ▼
┌───────────┐  ┌───────────────────────────────────┐
│ Workspace │  │          MemoryIndex               │
│           │  │  SQLite + FTS5 (BM25 keyword)      │
│ MEMORY.md │  │  + optional vector store           │
│ USER.md   │  │  hybrid re-ranking + MMR            │
│ AGENTS.md │  └──────────────┬────────────────────┘
│ RELATIONS │                 │
│ daily/    │  ┌──────────────▼────────────────────┐
└───────────┘  │       EmbeddingProvider            │
               │  NullProvider  (BM25-only)         │
               │  SentenceTransformer  (local)      │
               │  OpenAICompatible  (HTTP API)      │
               └───────────────────────────────────┘
```

**Memory files** live in the workspace directory as plain Markdown. They are human-readable and can be edited directly. The SQLite index is rebuilt automatically when files change.

**Bootstrap injection** assembles `MEMORY.md`, `USER.md`, `AGENTS.md`, `RELATIONS.md`, and recent daily logs into a single context block injected at session start.

**Compaction hooks** detect when the context window is filling up and emit structured prompts that instruct the agent to flush the conversation's important facts to storage before the window is summarized.

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
# Unit tests only (no embedding provider required — fast)
pytest tests/ -m "not embeddings"

# Integration tests (requires a configured embedding provider)
pytest tests/ -m embeddings

# Full suite
pytest tests/

# With coverage
pytest tests/ --cov=openmemory --cov-report=term-missing
```

The test suite has 108 tests: 99 unit tests and 9 integration tests. Integration tests are marked with `@pytest.mark.embeddings` and require an embedding provider to be configured via `openmemory.yaml` or environment variables.

### Submitting a PR

1. Fork the repository and create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes with accompanying tests.
3. Run `pytest tests/ -m "not embeddings"` — all unit tests must pass.
4. Open a pull request with a clear description of what changes and why.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.