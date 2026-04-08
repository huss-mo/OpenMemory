<img src="https://raw.githubusercontent.com/huss-mo/GroundMemory/master/_assets/icon.png" alt="om logo" width="140">

# GroundMemory

**Persistent identity and memory across AI tools - mcp-native, local-first, framework-agnostic, production-ready.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Release](https://github.com/huss-mo/GroundMemory/actions/workflows/pypi-publish.yml/badge.svg?event=push)](https://github.com/huss-mo/GroundMemory/actions/workflows/pypi-publish.yml)
[![Unit Tests](https://github.com/huss-mo/GroundMemory/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/huss-mo/GroundMemory/actions/workflows/unit-tests.yml)
[![Test Suite](https://img.shields.io/badge/test%20suite-509%20tests-blue.svg)](#running-the-test-suite)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![GitHub repo size](https://img.shields.io/github/repo-size/huss-mo/GroundMemory)
![GitHub language count](https://img.shields.io/github/languages/count/huss-mo/GroundMemory)
![GitHub top language](https://img.shields.io/github/languages/top/huss-mo/GroundMemory)

---

## Quick Start

### Option 1 - Docker (Recommended)

```bash
git clone https://github.com/huss-mo/GroundMemory && cd GroundMemory
docker compose up -d
# -> listening on http://127.0.0.1:4242/mcp
```

### Option 2 - pip

```bash
pip install groundmemory && groundmemory-mcp
# -> listening on http://127.0.0.1:4242/mcp
```

### Connect your client to the MCP server

```json
{
  "mcpServers": {
    "GroundMemory": {
      "url": "http://127.0.0.1:4242/mcp"
    }
  }
}
```

You can enable network access and replace `127.0.0.1` with your server's LAN IP - see [DOCS.md - Network Access](DOCS.md#network-access). </br>
You can also use the MCP server over the `stdio` transport - see [DOCS.md - Client Configuration](DOCS.md#client-configuration).

Your agent now has structured, searchable memory that persists across every session - long-term facts, a user profile, agent instructions, an entity graph, and daily logs - all managed automatically. No changes to your agent's code required.

This config works with any MCP-compatible client, including AI coding assistants (Cursor, Cline, Windsurf, Claude Code, Codex CLI), AI desktop clients (Claude Desktop, Open WebUI), and agent frameworks and platforms (LangChain, CrewAI, AutoGen, Google ADK, LiteLLM, n8n).

For installation options, embedding providers, multiple workspaces, and the Python API, see [DOCS.md](DOCS.md).

---

## What Becomes Possible

Without memory, every session in every AI tool starts from zero. With GroundMemory, agents can maintain continuity across time, accumulate knowledge, and behave like they actually know the person they're working with. It makes conversations stateful, fluid, and natural.

The exchange below is an example of what this looks like. The first session happens in **Open WebUI**. The second session happens in **Cline (VS Code)** - a completely different tool, with no shared history other than the GroundMemory workspace both are connected to.

---

#### Session 1 - Open WebUI (first ever session)

> **User**
>
> Hello

> **Agent**
>
> Hey! Welcome - this is the start of something good. I have persistent memory, so anything we talk about I'll carry with me into every future session. No need to repeat yourself down the line.
>
> A couple of quick things to get us started:
> - What should I call you?
> - Anything you always want me to keep in mind - how you like to work, things to avoid, preferences?
>
> No rush. Just getting to know you.

> **User**
>
> I'm Alice, a backend engineer working on a SaaS product. We're building a task management API - Node.js, PostgreSQL, Redis for caching, and we're planning to use BullMQ for job queues.

> **Agent**
>
> Got it, Alice. Backend engineer, task management API - Node.js, PostgreSQL, Redis, BullMQ for queues. I'll keep that as the baseline for everything we work on together.
>
> Anything else before we dive in? Or are you ready to get into it?

*The agent writes what it learned to memory.*

---

#### Session 2 - Cline in VS Code (different tool, same GroundMemory workspace)

> **User**
>
> Hey. What tech stack did we agree to use for the task management API?

> **Agent**
>
> You're building a task management API with Node.js, PostgreSQL, Redis for caching, and BullMQ for job queues. That's what we landed on.

*No introduction. No setup. The agent already knows.*

---


**A single identity across every AI tool you use.** Your memory workspace is not bound to one assistant. Connect Claude Desktop, Cursor, Cline, and any other MCP-compatible tool to the same GroundMemory server and they all share the same memory - your preferences, your stack, your ongoing work. You stop being a stranger every time you open a different tool. There is something genuinely different about being *known* rather than just answered - it shifts the relationship from transactional to collaborative, and removes the quiet tax of re-establishing context that most people don't notice until it's gone.

**A coding/personal assistant that builds a profile over time.** After a few conversations, it knows your schedule, your priorities, how you like to communicate, your tech stack, your preferred patterns, the architectural decisions you've already made. It doesn't need to ask.

**A research agent that constructs a knowledge graph.** As it reads papers and sources across many sessions, it records entities, relationships, and findings.

**A customer-facing agent with per-user memory.** In multi-user setups, each user gets their own workspace - preferences, history, ongoing context - giving every interaction a personalised, stateful feel without any custom infrastructure.

**A long-running autonomous agent that survives context limits.** Each new session calls `memory_bootstrap` to reload persisted facts, so the agent picks up exactly where the last one left off.

---

## What GroundMemory Does

Most agents are stateless. They ask the same questions again, repeat the same mistakes, and lose track of the user's preferences and ongoing work. This is not a model limitation - it is missing infrastructure.

GroundMemory provides that infrastructure. It gives your agent a structured, searchable memory that persists across sessions, organised into distinct tiers with clear ownership:

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
- **Zero-setup mode** - with `provider: none`, GroundMemory runs entirely on SQLite with FTS5. No API key, no GPU, no extra dependencies.
- **Pluggable embedding providers** - swap between a local sentence-transformers model, any OpenAI-compatible endpoint (OpenAI, Ollama, LM Studio, LiteLLM), or BM25-only without touching your agent code.
- **Workspace isolation** - each project, user, or agent gets its own directory-backed workspace with independent memory, relations, and daily logs.
- **Relation graph with semantic deduplication** - the graph automatically suppresses near-duplicate triples using configurable cosine similarity thresholding.

---

## How GroundMemory Compares

Comparison reflects publicly documented features as of MAR-2026. Submit a PR if anything is inaccurate.

| Feature | GroundMemory | Mem0 | Letta | memsearch | Zep |
|---|:---:|:---:|:---:|:---:|:---:|
| Zero-setup (no API key, no GPU) | ✅ | - | - | - | - |
| Local-first / offline | ✅ | - | - | Partial¹ | - |
| Human-readable Markdown memory | ✅ | - | - | ✅ | - |
| Structured memory tiers | ✅ | ✅² | ✅³ | - | - |
| Hybrid BM25 + vector search | ✅ | - | - | ✅ | ✅ |
| Entity relation graph | ✅ | ✅ | - | - | ✅ |
| MCP-native server | ✅ | Partial⁴ | Partial⁵ | - | - |
| Temporal knowledge graph | -⁶ | - | - | - | ✅ |
| Full agent framework | - | - | ✅ | - | - |
| Managed cloud service | - | ✅ | ✅ | - | ✅ |

¹ memsearch supports local ONNX embeddings + Milvus Lite, but requires initial model download </br>
² Mem0 organizes memory into Conversation, Session, User, and Organizational layers </br>
³ Letta uses Core Memory blocks (in-context) + Archival Memory (vector DB) + Conversation Search </br>
⁴ Mem0 offers an MCP integration but the primary interface is the Python/Node SDK </br>
⁵ Letta agents can consume external MCP servers as tools; Letta itself is not an MCP server </br>
⁶ GroundMemory timestamps all relations but does not support date-range queries.

---

## Tools

In normal mode, GroundMemory exposes four tools: `memory_bootstrap`, `memory_read`, `memory_write`, and `memory_relate`. An optional `memory_list` tool can be enabled via config. In dispatcher mode, all actions are routed through a single `memory_tool` call - useful for clients that perform better with fewer tools in scope.

**When using the MCP server**, instruct your agent to call `memory_bootstrap` at the start of every session before doing anything else, if you find out that it doesn't do that by default. This loads the full memory context (MEMORY.md, USER.md, AGENTS.md, RELATIONS.md, daily logs) into the conversation. Clients that support the MCP Prompts primitive (Cline, Claude Desktop) can instead use the `memory_bootstrap_prompt` prompt from their Prompts panel.

**When using the Python API**, call `session.bootstrap()` and pass the result as your system prompt - no tool call is needed.

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

GroundMemory is designed around three values:

1. **Simplicity over features.** Every addition must justify its complexity. A zero-dependency BM25-only mode must always work.
2. **Offline-first.** The default configuration must not require an API key, a network connection, or a GPU.
3. **Test-driven.** New behaviour ships with tests. The full suite must pass before any PR is merged.

### Development Setup

```bash
git clone https://github.com/huss-mo/GroundMemory.git
cd GroundMemory

# Install with all dev dependencies
pip install -e ".[dev,local]"

# Or with uv
uv sync --extra dev --extra local
```

### Running the Test Suite

```bash
# Core tests - no extra deps or config required (fast, always passes)
pytest

# Local model tests - sentence-transformers + cross-encoder
# Requires: pip install groundmemory[local]
pytest -m local

# API embedding tests - OpenAI-compatible HTTP endpoint
# Requires: endpoint configured via .env or groundmemory.yaml (see below)
pytest -m api_embeddings

# All marked tests together
pytest -m "local or api_embeddings"

# With coverage
pytest --cov=groundmemory --cov-report=term-missing
```

**`local` tests** download real sentence-transformers and cross-encoder models on first run.
Model names are read from config (`embedding.local_model`, `search.rerank_model`).
They skip automatically when `sentence-transformers` is not installed.

**`api_embeddings` tests** require a configured OpenAI-compatible embedding endpoint.
All settings are read from `.env` or `groundmemory.yaml` - whichever is found first
(`.env` takes priority). The tests skip automatically when `embedding.provider` is
not `openai` or the endpoint is unreachable.

Minimal `.env` for `api_embeddings` (project root or `~/.groundmemory/`):

```env
GROUNDMEMORY_EMBEDDING__PROVIDER=openai
GROUNDMEMORY_EMBEDDING__BASE_URL=http://localhost:11434/v1
GROUNDMEMORY_EMBEDDING__API_KEY=ollama
GROUNDMEMORY_EMBEDDING__MODEL=nomic-embed-text
```

Or equivalently via `groundmemory.yaml`:

```yaml
embedding:
  provider: openai
  base_url: http://localhost:11434/v1
  api_key: ollama
  model: nomic-embed-text
```

Any OpenAI-compatible endpoint works: OpenAI, Ollama, LM Studio, LiteLLM, etc.
Running `pytest` with no `-m` flag runs everything - marked tests skip gracefully
when their requirements are not met.

### Submitting a PR

1. Fork the repository and create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes with accompanying tests.
3. Run tests - all unit tests must pass.
4. Open a pull request with a clear description of what changes and why.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.