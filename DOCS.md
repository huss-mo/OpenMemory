# OpenMemory - Documentation

This document covers installation, integration guides, configuration reference, and environment variables.
For a project overview and quick start, see [README.md](README.md).

---

## Table of Contents

- [OpenMemory - Documentation](#openmemory---documentation)
  - [Table of Contents](#table-of-contents)
  - [Installation \& Configuration](#installation--configuration)
    - [Option 1 - Docker](#option-1---docker)
    - [Option 2 - pip](#option-2---pip)
    - [Embedding Providers](#embedding-providers)
  - [MCP Server](#mcp-server)
    - [Running the Server](#running-the-server)
    - [Client Configuration](#client-configuration)
    - [Available MCP Tools](#available-mcp-tools)
    - [Bootstrap - Loading Memory at Session Start](#bootstrap---loading-memory-at-session-start)
  - [Connecting to Your AI Agent Using The Python API](#connecting-to-your-ai-agent-using-the-python-api)
    - [OpenAI](#openai)
    - [Anthropic](#anthropic)
  - [Python API Example](#python-api-example)
  - [Tools Reference](#tools-reference)
  - [Architecture](#architecture)
    - [Architectural Layers](#architectural-layers)
      - [1. Workspace (`openmemory/core/workspace.py`)](#1-workspace-openmemorycoreworkspacepy)
      - [2. Memory Storage (`openmemory/core/storage.py`)](#2-memory-storage-openmemorycorestoragepy)
      - [3. Text Chunker (`openmemory/core/chunker.py`)](#3-text-chunker-openmemorycorechunkerpy)
      - [4. Embedding Providers (`openmemory/core/embeddings.py`)](#4-embedding-providers-openmemorycoreembeddingspy)
      - [5. Memory Index (`openmemory/core/index.py`)](#5-memory-index-openmemorycoreindexpy)
      - [6. Hybrid Search (`openmemory/core/search.py`)](#6-hybrid-search-openmemorycoresearchpy)
      - [7. Relation Graph (`openmemory/core/relations.py`)](#7-relation-graph-openmemorycorerelationspy)
      - [8. Sync (`openmemory/core/sync.py`)](#8-sync-openmemorycoresyncpy)
      - [9. Bootstrap Injector (`openmemory/bootstrap/injector.py`)](#9-bootstrap-injector-openmemorybootstrapinjectorpy)
      - [10. Compaction Hooks (`openmemory/bootstrap/compaction.py`)](#10-compaction-hooks-openmemorybootstrapcompactionpy)
      - [11. Tools (`openmemory/tools/`)](#11-tools-openmemorytools)
      - [12. LLM Adapters (`openmemory/adapters/`)](#12-llm-adapters-openmemoryadapters)
      - [13. Session (`openmemory/session.py`)](#13-session-openmemorysessionpy)
      - [13 (note). Session vs Workspace - not two different things](#13-note-session-vs-workspace---not-two-different-things)
  - [Data Flow](#data-flow)
  - [Tech Stack](#tech-stack)
  - [Configuration](#configuration)
    - [Minimum Config](#minimum-config)
    - [openmemory.yaml Reference](#openmemoryyaml-reference)
    - [Environment Variables](#environment-variables)

---

## Installation & Configuration

### Option 1 - Docker

Docker is the recommended way to run OpenMemory. It requires no Python environment setup and keeps your workspace data in a local `./data` directory.

```bash
git clone https://github.com/huss-mo/OpenMemory
cd OpenMemory
cp openmemory/config/.env.example .env
docker compose up -d
# → listening on http://0.0.0.0:4242/mcp
```

The default compose file starts a single `openmemory` service using BM25-only search (no embedding API required). Edit `.env` to switch providers - see [Embedding Providers](#embedding-providers) below.

To run a **second workspace** on a different port (e.g. for a separate project or user), uncomment the `openmemory-personal` service in `docker-compose.yml`:

```yaml
# openmemory-personal:
#   build:
#     context: .
#   image: openmemory:latest
#   restart: unless-stopped
#   ports:
#     - "4243:4242"
#   volumes:
#     - ./data:/data
#   env_file:
#     - .env
#   environment:
#     OPENMEMORY_WORKSPACE: personal
```

Workspace data is stored in `./data/<workspace-name>/` on the host and persists across container restarts.

**Building with sentence-transformers (local embeddings)**

The default Docker image does not include `sentence-transformers`. To build an image that supports the `local` embedding provider, pass the `EXTRAS=local` build argument:

```bash
docker compose build --build-arg EXTRAS=local
docker compose up -d
```

Then set `OPENMEMORY_EMBEDDING__PROVIDER=local` in your `.env`.

### Option 2 - pip

For development or direct integration without Docker:

```bash
git clone https://github.com/huss-mo/OpenMemory
cd OpenMemory

# BM25-only - no extra dependencies
pip install -e .

# With local sentence-transformers embeddings
pip install -e ".[local]"

# Or with uv
uv sync
uv sync --extra local   # for sentence-transformers support
```

Then start the MCP server:

```bash
OPENMEMORY_WORKSPACE=my-project openmemory-mcp
# → listening on http://0.0.0.0:4242/mcp
```

**Configuration for pip installs**

OpenMemory reads config from `~/.openmemory/` - the same directory where workspace data lives. Place your config files there once and they'll be found regardless of which directory you run `openmemory-mcp` from:

```
~/.openmemory/
├── .env                  ← environment-variable style config
├── openmemory.yaml       ← YAML style config
└── default/              ← workspace data (auto-created on first run)
    ├── MEMORY.md
    └── ...
```

Both `.env` and `openmemory.yaml` are optional - use whichever format you prefer (or neither, and set env vars directly). Environment variables always take priority over config files.

On first run, `openmemory-mcp` automatically copies an annotated example config into `~/.openmemory/openmemory.yaml.example` - the full YAML reference with every option documented. You can also find it in the repository at `openmemory/config/openmemory.yaml.example`.

For environment-variable style config, copy the bundled example manually:

```bash
cp openmemory/config/.env.example ~/.openmemory/.env
```

**A cwd-level config file (`./openmemory.yaml` or `./.env`) is also checked** as a fallback, which is useful for per-project overrides in dev mode.

### Embedding Providers

OpenMemory supports three embedding providers. You can switch between them at any time by changing `OPENMEMORY_EMBEDDING__PROVIDER` (or `embedding.provider` in `openmemory.yaml`). No data migration is required.

| Provider | Value | Extra install required? | When to use |
|---|---|---|---|
| BM25-only | `none` | No | Default. Pure keyword search via SQLite FTS5. Works offline, no API key needed. |
| OpenAI-compatible API | `openai` | No | Any HTTP embedding API: OpenAI, Ollama, LM Studio, LiteLLM, vLLM, Mistral, etc. Requires `OPENMEMORY_EMBEDDING__BASE_URL` and `OPENMEMORY_EMBEDDING__API_KEY`. |
| Local sentence-transformers | `local` | Yes - `pip install -e ".[local]"` or `--build-arg EXTRAS=local` for Docker | Fully offline vector embeddings. Downloads model on first run. |

**`none` - BM25-only (default)**

No configuration needed. OpenMemory uses SQLite FTS5 for all search. Ideal for getting started quickly or for air-gapped environments.

```bash
OPENMEMORY_EMBEDDING__PROVIDER=none openmemory-mcp
```

**`openai` - OpenAI-compatible HTTP API**

Works with any endpoint that follows the OpenAI embeddings API format. No extra Python packages are required - only `httpx`, which is a core dependency already installed with OpenMemory.

```bash
# Real OpenAI
OPENMEMORY_EMBEDDING__PROVIDER=openai \
OPENMEMORY_EMBEDDING__API_KEY=sk-... \
OPENMEMORY_EMBEDDING__MODEL=text-embedding-3-small \
openmemory-mcp

# Ollama (local server, no API key needed)
OPENMEMORY_EMBEDDING__PROVIDER=openai \
OPENMEMORY_EMBEDDING__BASE_URL=http://localhost:11434/v1 \
OPENMEMORY_EMBEDDING__API_KEY=ollama \
OPENMEMORY_EMBEDDING__MODEL=nomic-embed-text \
openmemory-mcp

# LM Studio
OPENMEMORY_EMBEDDING__PROVIDER=openai \
OPENMEMORY_EMBEDDING__BASE_URL=http://localhost:1234/v1 \
OPENMEMORY_EMBEDDING__API_KEY=lm-studio \
OPENMEMORY_EMBEDDING__MODEL=nomic-ai/nomic-embed-text-v1.5-GGUF \
openmemory-mcp
```

**`local` - sentence-transformers (offline)**

Runs a local embedding model entirely on your machine - no network call, no API key. Requires installing the optional `local` extra, which pulls in `sentence-transformers` and its dependencies (PyTorch, Transformers, etc.). The model is downloaded from HuggingFace on first use.

```bash
# Install the extra first
pip install -e ".[local]"

OPENMEMORY_EMBEDDING__PROVIDER=local \
OPENMEMORY_EMBEDDING__LOCAL_MODEL=all-MiniLM-L6-v2 \
openmemory-mcp
```

---

## MCP Server

OpenMemory can run as a standalone MCP (Model Context Protocol) server over HTTP, exposing all 8 memory tools to any MCP-compatible client - including Claude Desktop, Cursor, Cline, and custom agents.

Each server instance owns a single workspace. Multiple workspaces require multiple server processes running on different ports.

### Running the Server

The `openmemory-mcp` command is available after installing OpenMemory (MCP support is included by default).

```bash
# Default: workspace "default", host 0.0.0.0, port 4242
openmemory-mcp

# Custom workspace
OPENMEMORY_WORKSPACE=my-project openmemory-mcp

# Custom host and port
OPENMEMORY_MCP_HOST=127.0.0.1 OPENMEMORY_MCP_PORT=9000 openmemory-mcp
```

The server starts at `http://<host>:<port>/mcp` using the `streamable-http` MCP transport.

### Client Configuration

OpenMemory speaks standard MCP over HTTP, so any MCP-compatible client works. The table below shows the most common categories and examples — the list is not exhaustive.

| Category | Examples | How to connect |
|---|---|---|
| AI coding assistants | Cursor, Cline, Windsurf, Claude Code, Codex CLI | Add the JSON snippet below to your client's MCP server config |
| AI desktop clients | Claude Desktop, Open WebUI | Add the JSON snippet below via the client's settings |
| Agent frameworks & platforms | LangChain, LangGraph, CrewAI, AutoGen, Google ADK, LiteLLM, n8n | Python API (`MemorySession`) or point an HTTP tool node at the MCP endpoint |

```json
{
  "mcpServers": {
    "openmemory": {
      "url": "http://<server-ip>:4242/mcp"
    }
  }
}
```

Clients that support the MCP Prompts primitive (such as Cline and Claude Desktop) will also show a `memory_bootstrap_prompt` entry in their Prompts panel — click it at session start to inject memory context without waiting for the agent to call the tool. For agent frameworks and platforms that use the Python API, see [Connecting to Your AI Agent Using The Python API](#connecting-to-your-ai-agent-using-the-python-api).

### Available MCP Tools

Once connected, the client has access to 9 memory tools and 1 prompt:

| Tool | Description |
|---|---|
| `memory_bootstrap` | **Call once at session start.** Returns the full memory context (MEMORY.md, USER.md, AGENTS.md, RELATIONS.md, daily logs) as a formatted string. |
| `memory_write` | Store a memory in long-term storage (`MEMORY.md`) or today's daily log |
| `memory_search` | Hybrid semantic + keyword search across all memory tiers |
| `memory_get` | Read a slice of a workspace file by line range |
| `memory_list` | List workspace files or preview a specific file |
| `memory_delete` | Delete a 1-indexed line range from a mutable workspace file |
| `memory_replace_text` | Replace the first occurrence of an exact string in a mutable workspace file |
| `memory_replace_lines` | Replace a 1-indexed inclusive line range in a mutable workspace file |
| `memory_relate` | Record a typed entity relationship (`subject → predicate → object`) |

| Prompt | Description |
|---|---|
| `memory_bootstrap_prompt` | Same content as `memory_bootstrap`, exposed as an MCP Prompt for clients that support the Prompts primitive (Cline, Claude Desktop). Click it in your client's Prompts panel at session start instead of waiting for the agent to call the tool. |

### Bootstrap - Loading Memory at Session Start

OpenMemory's memory context (long-term facts, user profile, agent instructions, entity graph, daily logs) needs to be loaded at the start of each session. Two mechanisms are provided:

**Tool-based bootstrap (all clients)**

The `memory_bootstrap` tool description is written so that most agents call it automatically at the start of a session without any explicit instruction - the tool's description alone signals that it should be the first action taken. **No system-prompt changes are necessary in most cases.**

If you find that your agent does not call `memory_bootstrap` on its own, you can add an explicit fallback instruction to the system prompt:

```
At the start of every session, call memory_bootstrap before doing anything else.
Use the returned context as your background knowledge for the rest of the session.
```

**Prompt-based bootstrap (Cline, Claude Desktop)**

Clients that support the MCP Prompts primitive (Cline, Claude Desktop) will show a `memory_bootstrap_prompt` entry in their Prompts panel. Click it at the start of a session to inject the memory context directly into the conversation - no agent tool call required. This is an alternative to the tool-based path, useful when you want to load memory context manually rather than waiting for the agent to call the tool.

The content returned by both mechanisms is identical.

---

## Connecting to Your AI Agent Using The Python API

OpenMemory exposes standard JSON schemas for function calling, compatible with OpenAI and Anthropic out of the box. The primary export is `ALL_TOOLS` - a list of `(schema, run)` pairs. Pass the schemas to the model so it knows what tools are available; when the model calls a tool, dispatch it back through the paired `run` function (or use `session.execute_tool` directly). Both paths are shown below.

### OpenAI

```python
from openai import OpenAI
from openmemory.session import MemorySession
from openmemory.tools import ALL_TOOLS
import json

session = MemorySession.create("my-project")
client = OpenAI()

# Build the tools list for the API call
tools = [{"type": "function", "function": schema} for schema, _ in ALL_TOOLS]

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": session.bootstrap()},
        {"role": "user", "content": "What do you remember about my preferences?"},
    ],
    tools=tools,
)

# Dispatch tool calls the model makes back to OpenMemory
for call in response.choices[0].message.tool_calls or []:
    result = session.execute_tool(call.function.name, **json.loads(call.function.arguments))
    print(result)
```

### Anthropic

```python
from anthropic import Anthropic
from openmemory.session import MemorySession
from openmemory.tools import ALL_TOOLS

session = MemorySession.create("my-project")
client = Anthropic()

# Anthropic tool schema uses input_schema instead of parameters
tools = [
    {
        "name": schema["name"],
        "description": schema["description"],
        "input_schema": schema["parameters"],
    }
    for schema, _ in ALL_TOOLS
]

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    system=session.bootstrap(),
    messages=[{"role": "user", "content": "Summarize what you know about me."}],
    tools=tools,
)

# Dispatch tool use blocks back to OpenMemory
for block in response.content:
    if block.type == "tool_use":
        result = session.execute_tool(block.name, **block.input)
        print(result)
```

> **Complete runnable agents** - `examples/openai_agent.py` and `examples/anthropic_agent.py`
> show a full interactive loop with workspace sync on startup, compaction detection, and
> graceful shutdown. Run them with:
> ```bash
> uv run python examples/openai_agent.py
> uv run python examples/anthropic_agent.py
> ```

---

## Python API Example

```python
from openmemory.session import MemorySession

# Create (or reopen) a named workspace
session = MemorySession.create("my-project")

# Write a long-term memory
session.execute_tool("memory_write", content="User prefers concise answers.", tier="long_term")

# Write a daily log entry
session.execute_tool("memory_write", content="Working on the auth service refactor.", tier="daily")

# Record a relationship between entities
session.execute_tool("memory_relate", subject="Alice", predicate="works_at", object="Acme Corp")

# Search across all memory tiers
result = session.execute_tool("memory_search", query="communication preferences")
for item in result["data"]["results"]:
    print(item["content"])

# Build the system prompt context block for your agent
system_prompt = session.bootstrap()
```

`MemorySession.create("my-project")` creates a workspace directory at `~/.openmemory/my-project/` on first run, seeding all required files. Subsequent calls reopen the same workspace.

---

## Tools Reference

Use `session.execute_tool(name, **kwargs)` to call tools programmatically, or pass `ALL_TOOLS` to your model framework directly.

> **Immutability rule:** `MEMORY.md` and all `daily/*.md` files are append-only. `memory_delete`, `memory_replace_text`, and `memory_replace_lines` will reject any attempt to modify them. Use `memory_write` to append new information to these files instead.

| Tool | Description | Required Parameters | Optional Parameters |
|---|---|---|---|
| `memory_write` | Append a memory to long-term storage or today's daily log | `content` | `tier` (default: `long_term`) |
| `memory_search` | Hybrid semantic + keyword search across all memory tiers | `query` | `top_k`, `source` |
| `memory_get` | Read a slice of a workspace file by 1-indexed line range | `file` | `start_line`, `end_line` |
| `memory_list` | List workspace files or preview a specific file | - | `file` |
| `memory_delete` | Tombstone-delete a 1-indexed line range from a mutable file | `file`, `start_line`, `end_line` | - |
| `memory_replace_text` | Replace the first exact string match in a mutable file | `file`, `search`, `replacement` | - |
| `memory_replace_lines` | Replace a 1-indexed inclusive line range in a mutable file | `file`, `start_line`, `end_line`, `replacement` | - |
| `memory_relate` | Record a typed entity relationship (`subject → predicate → object`) | `subject`, `predicate`, `object` | - |

**`memory_write` tiers** (`tier` parameter):

| Value | Written to | Behaviour |
|---|---|---|
| `long_term` | `MEMORY.md` | Appended permanently; survives all sessions. **Immutable** - append only. |
| `daily` | `daily/YYYY-MM-DD.md` | Appended to today's date-stamped log. **Immutable** - append only. |
| `user` | `USER.md` | Updates the stable user profile. Mutable. |
| `agent` | `AGENTS.md` | Updates agent operating instructions. Mutable. |

**`memory_search` source filters** (`source` parameter):

| Value | Searches |
|---|---|
| *(omitted)* | All tiers |
| `long_term` | `MEMORY.md` only |
| `daily` | All daily logs |
| `relations` | `RELATIONS.md` + SQLite graph |
| `user` | `USER.md` only |
| `agents` | `AGENTS.md` only |

---

## Architecture

### Architectural Layers

#### 1. Workspace (`openmemory/core/workspace.py`)
`Workspace` is a pure filesystem abstraction for a single memory workspace. On first use it creates the directory tree and seeds the default Markdown files (`MEMORY.md`, `USER.md`, `AGENTS.md`, `RELATIONS.md`). All other layers receive a `Workspace` object to resolve file paths - it never holds runtime state such as a database connection or embedding provider.

The on-disk layout is a **single directory level** under `~/.openmemory`:

```
~/.openmemory/
└── <workspace_name>/          ← one directory per named workspace
    ├── MEMORY.md              long-term curated memory
    ├── USER.md                stable user profile
    ├── AGENTS.md              agent operating instructions
    ├── RELATIONS.md           human-readable entity relation graph
    ├── daily/
    │   └── YYYY-MM-DD.md     append-only daily logs
    └── .index/
        └── memory.db          SQLite index (chunks + FTS5 + relations + embeddings)
```

#### 2. Memory Storage (`openmemory/core/storage.py`)
Low-level atomic Markdown file I/O. All writes go through a temp-file + rename cycle to prevent partial writes. Provides `write_long_term`, `write_daily`, `read_file`, `delete_lines`, and `list_daily_files`.

#### 3. Text Chunker (`openmemory/core/chunker.py`)
Splits Markdown files into overlapping chunks that respect heading boundaries. Each `Chunk` carries a deterministic `chunk_id` (SHA-256 of path + start line + text) and 0-indexed line ranges for precise `memory_get` retrieval.

#### 4. Embedding Providers (`openmemory/core/embeddings.py`)
Abstract `EmbeddingProvider` with three concrete implementations:

| Provider | Class | Extra install | When to use |
|---|---|---|---|
| `none` | `NullEmbeddingProvider` | None | Zero-dep BM25-only mode - returns empty vectors; no API key or GPU needed |
| `openai` | `OpenAICompatibleProvider` | None (`httpx` is a core dep) | Any OpenAI-compatible HTTP endpoint (OpenAI, Ollama, LM Studio, LiteLLM, …) |
| `local` | `SentenceTransformerProvider` | `pip install -e ".[local]"` | Fully offline embeddings via `sentence-transformers`; model downloaded on first run |

#### 5. Memory Index (`openmemory/core/index.py`)
SQLite database (`memory.db`) with five tables:

| Table | Purpose |
|---|---|
| `files` | Tracks indexed files with SHA-256 hash + mtime for change detection |
| `chunks` | Text chunks with JSON-serialised embedding vectors |
| `chunks_fts` | FTS5 virtual table - BM25 keyword search via SQLite triggers |
| `relations` | Named entity relationships (subject → predicate → object) |
| `embedding_cache` | Reuses embeddings when chunk content hasn't changed |

Vector search is implemented in pure Python (NumPy cosine similarity) so it works everywhere without native extensions. The database runs in WAL (Write-Ahead Logging) mode (`PRAGMA journal_mode=WAL`) for better concurrent read performance.

#### 6. Hybrid Search (`openmemory/core/search.py`)
Seven-step pipeline:
1. **Embed** the query via the configured provider.
2. **Vector search** - cosine similarity over all chunk embeddings → top `k × candidate_multiplier` candidates.
3. **Keyword search** - FTS5 BM25 → top `k × candidate_multiplier` candidates.
4. **Merge & re-score** - `score = vector_weight × vec_score + (1 − vector_weight) × bm25_score`.
5. **Temporal decay** - `score × exp(−decay_rate × days_old)` (disabled by default).
6. **Graph expansion** - extract entity mentions from top results, attach related relation triples as `relation_context`.
7. Return top `k` as `SearchResult` objects.

#### 7. Relation Graph (`openmemory/core/relations.py`)
Single consolidated module for all relation logic. Stores typed entity triples (`subject → predicate → object`) in two places simultaneously:
- **SQLite** `relations` table - fast structured lookup used by graph expansion during search.
- **`RELATIONS.md`** - human-readable, editable mirror, injected at bootstrap.

**Source-of-truth model:** `RELATIONS.md` is the authoritative record. Any change to the file is automatically reconciled back into SQLite.

**Public API:**

| Symbol | Description |
|---|---|
| `add_relation(...)` | Write a relation to both SQLite and RELATIONS.md with semantic dedup |
| `get_relations(...)` | Read relations from SQLite |
| `parse_relations_from_text(text)` | Parse valid relation lines from a raw string; used by `memory_delete` to identify rows to remove without a temp-file round-trip |
| `parse_relations_from_file(path)` | Parse valid lines from RELATIONS.md into `{subject, predicate, object, note}` dicts (delegates to `parse_relations_from_text`) |
| `sync_relations_from_file(path, index)` | Upsert/delete SQLite rows to match RELATIONS.md exactly; called by `sync_file` / `sync_workspace` and via `sync_after_edit` after every in-place edit |
| `validate_relations_replacement(text)` | Validate that every non-blank, non-comment line in a replacement string matches the RELATIONS.md format; returns `(all_valid, valid_lines, invalid_lines)` |
| `format_relations_for_context` | Format the relation graph as a Markdown block for bootstrap injection |
| `RELATION_LINE_RE` | Compiled regex for a valid RELATIONS.md line |
| `RELATIONS_FORMAT_REMINDER` | Human-readable format reminder string included in validation error responses |

Format for each line in RELATIONS.md:
```
- [Subject] --predicate--> [Object] (YYYY-MM-DD) — "optional note"
```

Semantic deduplication: before inserting, the new triple is embedded and compared (cosine similarity) against all existing triples. If similarity ≥ `dedup_threshold` (default 0.92) the write is skipped and the existing triple is returned.

#### 8. Sync (`openmemory/core/sync.py`)
Keeps the SQLite index consistent with the Markdown files using SHA-256 content hashing (not timestamps). `sync_workspace` walks all files and re-indexes changed ones. `sync_file` force-syncs a single file - called immediately after every `memory_write` so new content is searchable within the same session.

#### 9. Bootstrap Injector (`openmemory/bootstrap/injector.py`)
Assembles a system-prompt block from workspace files, respecting per-file and total character budgets (`max_chars_per_file`, `max_total_chars`). Truncated files get a visible `[TRUNCATED - use memory_get to read the rest]` marker. Injects (in order): long-term memory, user profile, agent instructions, relation graph, yesterday's and today's daily logs.

#### 10. Compaction Hooks (`openmemory/bootstrap/compaction.py`)
`should_flush(current_tokens, context_window, cfg)` returns `True` when the remaining context budget drops below the configured threshold. `get_compaction_prompts(cfg)` returns the `{system, user}` messages the agent uses to flush important facts to storage before the window is summarised.

> **Python API only.** Compaction hooks are only meaningful when you control the message loop yourself - i.e. when using the Python API directly (via `session.should_compact()` and `session.compaction_prompts()`). When OpenMemory is running as an MCP server, it has no visibility into the client's conversation history or token usage, so compaction cannot be triggered automatically. In that case, compaction is the responsibility of the MCP client or agent framework.

#### 11. Tools (`openmemory/tools/`)
Eight JSON-schema-described tools exposed to the LLM via function calling:

| Tool | File written | Notes |
|---|---|---|
| `memory_write` | `MEMORY.md` or `daily/YYYY-MM-DD.md` | Immediately re-indexes the changed file |
| `memory_search` | - | Full hybrid search pipeline |
| `memory_get` | - | Line-range read of any workspace file |
| `memory_list` | - | Directory listing or file preview |
| `memory_delete` | Mutable files only | Tombstone-style deletion (1-indexed); re-indexes. Rejected on `MEMORY.md`/`daily/*.md`. When file is `RELATIONS.md`, also deletes the corresponding SQLite relation rows via `parse_relations_from_text`. |
| `memory_replace_text` | Mutable files only | Replaces first exact string match in-place; re-indexes. Rejected on `MEMORY.md`/`daily/*.md`. When file is `RELATIONS.md`, validates replacement format via `validate_relations_replacement` and reconciles SQLite. |
| `memory_replace_lines` | Mutable files only | Replaces a 1-indexed inclusive line range in-place; re-indexes. Rejected on `MEMORY.md`/`daily/*.md`. When file is `RELATIONS.md`, validates replacement format via `validate_relations_replacement` and reconciles SQLite. |
| `memory_relate` | `RELATIONS.md` + SQLite | Semantic dedup before insert. Pass `supersedes=True` to replace all prior `(subject, predicate)` triples (e.g. job change, relocation). |

**Shared utilities (`openmemory/tools/base.py`):**

| Symbol | Description |
|---|---|
| `ok(data)` / `err(msg)` | Wrap a successful or error tool result |
| `is_immutable(file)` | Return `True` for `MEMORY.md` and `daily/*.md` |
| `sync_after_edit(session, resolved, is_relations, base_payload)` | Re-index a file after an in-place edit and return `ok(payload)`. Calls `sync_file` (and, when `is_relations=True`, `sync_relations_from_file`) non-fatally - sync failures add a `warning` key rather than raising. Used by `memory_delete`, `memory_replace_text`, and `memory_replace_lines` to eliminate the duplicated re-index block that each function would otherwise carry. |

#### 12. LLM Adapters (`openmemory/adapters/`)
Thin schema-conversion + agentic-loop helpers:
- **`adapters/openai.py`** - converts schemas to OpenAI function-calling format; `handle_tool_calls` dispatches tool calls and appends results to the message list; `run_agent_loop` iterates until the model stops calling tools.
- **`adapters/anthropic.py`** - same for Anthropic's `tool_use` / `tool_result` block format.

#### 13. Session (`openmemory/session.py`)
`MemorySession` is the composition root that holds references to `Workspace`, `MemoryIndex`, and `EmbeddingProvider`. It exposes `execute_tool`, `bootstrap`, `sync`, `should_compact`, and `compaction_prompts` as the primary API surface.

#### 13 (note). Session vs Workspace - not two different things
`MemorySession` **is** the workspace session - there is no meaningful distinction between the two concepts in OpenMemory. `Workspace` is the low-level filesystem handle; `MemorySession` is the high-level runtime object that wraps it together with the index and embedding provider. When you call `MemorySession.create("my-project")`, it resolves the path as `~/.openmemory/my-project` - a single directory, not a nested one.

---

## Data Flow

```
User message
     │
     ▼
MemorySession.bootstrap()
     │  Reads MEMORY.md, USER.md, AGENTS.md, RELATIONS.md, daily logs
     │  → assembled into system prompt block
     ▼
LLM receives system prompt + tool schemas + user message
     │
     │  Model may call memory tools:
     │
     ├─► memory_write(content, tier)
     │       └─► storage.write_long_term / write_daily  → appends to Markdown
     │       └─► sync.sync_file                         → chunk → embed → upsert SQLite
     │
     ├─► memory_search(query, top_k, source)
     │       └─► provider.embed(query)                  → query vector
     │       └─► index.vector_search                    → cosine top-k
     │       └─► index.keyword_search                   → FTS5 BM25 top-k
     │       └─► merge + decay + graph expansion        → ranked SearchResult list
     │
     ├─► memory_get(file, start_line, end_line)
     │       └─► storage.read_file                      → raw Markdown slice
     │
     ├─► memory_list(target, file)
     │       └─► workspace.all_memory_files / storage   → file listing / preview
     │
     ├─► memory_delete(file, start_line, end_line)
     │       └─► is_immutable(file) check               → reject if MEMORY.md or daily/*.md
     │       └─► storage.delete_lines                   → tombstone in Markdown
     │       └─► sync.sync_file                         → re-index
     │
     ├─► memory_replace_text(file, search, replacement)
     │       └─► is_immutable(file) check               → reject if MEMORY.md or daily/*.md
     │       └─► storage.replace_text                   → first-match replacement in Markdown
     │       └─► sync.sync_file                         → re-index
     │
     ├─► memory_replace_lines(file, start_line, end_line, replacement)
     │       └─► is_immutable(file) check               → reject if MEMORY.md or daily/*.md
     │       └─► storage.replace_lines                  → line-range replacement in Markdown
     │       └─► sync.sync_file                         → re-index
     │
     └─► memory_relate(subject, predicate, object)
             └─► relations._find_semantic_duplicate      → cosine dedup check
             └─► index.insert_relation                  → SQLite relations table
             └─► storage._atomic_write                  → append to RELATIONS.md
     │
     ▼
Agent response returned to user
     │
     (optionally)
     ▼
session.should_compact(current_tokens, context_window)
     │  True → inject compaction_prompts → agent flushes session to memory_write
     ▼
Next session: session.bootstrap() reloads persisted facts
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Configuration | Pydantic Settings + PyYAML (YAML file + env vars) |
| Database | SQLite via `sqlite3` stdlib, WAL mode (`PRAGMA journal_mode=WAL`) |
| Full-text search | SQLite FTS5 (BM25) with auto-sync triggers |
| Vector search | NumPy cosine similarity (pure Python; no native extension required) |
| Embeddings - local | `sentence-transformers` (optional extra; not installed by default) |
| Embeddings - remote | Any OpenAI-compatible HTTP endpoint via `httpx` (core dependency) |
| HTTP client | `httpx` |
| Packaging | `hatchling` build backend (`pyproject.toml`), installable via `uv` or `pip` |
| Tests | `pytest` |

---

## Configuration

### Minimum Config

No configuration file is required. With no config, OpenMemory uses BM25-only search backed by SQLite - no API key, no GPU, no extra packages.

**Finding the example config files**

Both example files are bundled with the package under `openmemory/config/` in the repository:

- `openmemory/config/openmemory.yaml.example` - full YAML reference with every option documented
- `openmemory/config/.env.example` - all environment variables with descriptions and defaults

For **pip installs**, `openmemory-mcp` automatically copies `openmemory.yaml.example` into `~/.openmemory/` on first run. For **Docker installs**, copy the `.env.example` manually as shown in the [Docker quick-start](#option-1---docker) above.

### openmemory.yaml Reference

**Config file search order (first match wins):**

| Location | Resolved path | Use case |
|---|---|---|
| `$OPENMEMORY_ROOT_DIR/openmemory.yaml` | `~/.openmemory/openmemory.yaml` (pip) or `/data/openmemory.yaml` → `./data/openmemory.yaml` on host (Docker) | Global user config — recommended for pip installs and Docker |
| `./openmemory.yaml` | cwd at process start | Per-project override in dev mode |

The same search order applies to `.env` files (`$OPENMEMORY_ROOT_DIR/.env` then `./.env`).

Settings in these files are overridden by environment variables, which in turn are overridden by constructor kwargs.

```yaml
# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------

# Root directory for all workspaces (default: ~/.openmemory)
# root_dir: ~/.openmemory

# Default workspace name
# workspace: default

# ---------------------------------------------------------------------------
# Embedding provider
# ---------------------------------------------------------------------------
embedding:
  # provider options:
  #   "none"   - BM25-only, no extra deps, no API key needed (default)
  #   "openai" - any OpenAI-compatible HTTP endpoint, no extra deps needed
  #   "local"  - sentence-transformers (install with: pip install -e ".[local]")
  provider: none

  # --- sentence-transformers (provider: local) ---
  # Requires: pip install -e ".[local]"  (not installed by default)
  # Any model from https://www.sbert.net/docs/pretrained_models.html
  # local_model: all-MiniLM-L6-v2      # fast, 384-dim, good general quality
  # local_model: all-mpnet-base-v2     # slower, 768-dim, higher quality

  # --- OpenAI-compatible API (provider: openai) ---
  # No extra install required. Supports OpenAI, Ollama, LM Studio, vLLM,
  # LiteLLM, Mistral, Together, and any endpoint following the OpenAI format.

  # Real OpenAI:
  # base_url: ~        # leave blank to use api.openai.com
  # api_key: sk-...
  # model: text-embedding-3-small

  # Ollama (local):
  # base_url: http://localhost:11434/v1
  # api_key: ollama          # required by the client but ignored by Ollama
  # model: nomic-embed-text  # pull with: ollama pull nomic-embed-text

  # LM Studio:
  # base_url: http://localhost:1234/v1
  # api_key: lm-studio
  # model: nomic-ai/nomic-embed-text-v1.5-GGUF

  # LiteLLM proxy:
  # base_url: http://localhost:4000/v1
  # api_key: sk-...
  # model: text-embedding-3-small

  # Number of texts sent per embedding API call
  # batch_size: 64

# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------
search:
  # Number of results returned per memory_search call
  top_k: 6

  # Oversampling factor: top_k * candidate_multiplier candidates are fetched
  # from each path (vector + keyword) before merging and re-ranking
  candidate_multiplier: 4

  # Weight for vector similarity score (0.0–1.0)
  # keyword_weight = 1.0 - vector_weight
  # Set to 0.0 for pure BM25 (recommended when provider: none)
  vector_weight: 0.7

  # Temporal decay: score *= exp(-decay_rate * days_old)
  # 0.0 disables decay; 0.01 halves relevance after ~70 days
  temporal_decay_rate: 0.0

  # MMR (Maximal Marginal Relevance) diversity re-ranking
  # 0.0 = disabled (pure relevance ranking), 1.0 = maximum result diversity
  mmr_lambda: 0.0

# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------
chunking:
  # Target chunk size in approximate tokens (1 token ≈ 4 chars)
  tokens: 400

  # Overlap between consecutive chunks in approximate tokens
  overlap: 80

# ---------------------------------------------------------------------------
# Relation graph
# ---------------------------------------------------------------------------
relations:
  # Cosine similarity threshold above which two relation triples are considered
  # semantic duplicates and merged. Set to 1.0 to disable semantic dedup.
  dedup_threshold: 0.92

# ---------------------------------------------------------------------------
# Bootstrap - system-prompt injection at session start
# ---------------------------------------------------------------------------
bootstrap:
  # Maximum characters per file before a truncation warning is appended
  max_chars_per_file: 20000

  # Maximum total characters injected across all files
  max_total_chars: 150000

  # Which memory files to inject into the system prompt
  inject_long_term_memory: true  # MEMORY.md
  inject_user_profile: true      # USER.md
  inject_agents: true            # AGENTS.md
  inject_daily_logs: true        # daily/YYYY-MM-DD.md (today + yesterday)
  inject_relations: true         # RELATIONS.md

  # Reconcile the SQLite relations table from RELATIONS.md at session start.
  # Enable when you edit RELATIONS.md manually outside the agent so that
  # changes are reflected at the next session start. Disabled by default.
  sync_relations_on_bootstrap: false

# ---------------------------------------------------------------------------
# Compaction - pre-context-window-flush hooks
# ---------------------------------------------------------------------------
compaction:
  # Enable compaction detection
  enabled: true

  # Remaining tokens in the context window that cause should_compact() to return True
  soft_threshold_tokens: 4000

  # Minimum tokens always kept free for the model's response
  reserve_floor_tokens: 20000

  # Messages injected at the flush turn (override to customise wording)
  # system_prompt: "Session nearing compaction. Store durable memories now."
  # user_prompt: "Review the conversation and write lasting facts to memory. Reply DONE when finished."

# ---------------------------------------------------------------------------
# MCP server (openmemory-mcp command)
# ---------------------------------------------------------------------------
# mcp:
#   # Host address the MCP server binds to
#   host: 0.0.0.0
#   # TCP port the MCP server listens on
#   port: 4242
```

---

### Environment Variables

All settings are available as environment variables using the `OPENMEMORY_` prefix. Nested keys use double-underscore (`__`) as a separator. Environment variables take priority over `openmemory.yaml`.

**Embedding**

| Variable | Description | Default |
|---|---|---|
| `OPENMEMORY_EMBEDDING__PROVIDER` | `none` / `local` / `openai` | `none` |
| `OPENMEMORY_EMBEDDING__BASE_URL` | OpenAI-compatible endpoint URL | - |
| `OPENMEMORY_EMBEDDING__API_KEY` | API key for the endpoint | - |
| `OPENMEMORY_EMBEDDING__MODEL` | Embedding model name (provider: openai) | `text-embedding-3-small` |
| `OPENMEMORY_EMBEDDING__LOCAL_MODEL` | sentence-transformers model name | `all-MiniLM-L6-v2` |
| `OPENMEMORY_EMBEDDING__BATCH_SIZE` | Texts per embedding API call | `64` |

**Search**

| Variable | Description | Default |
|---|---|---|
| `OPENMEMORY_SEARCH__TOP_K` | Results returned per query | `6` |
| `OPENMEMORY_SEARCH__CANDIDATE_MULTIPLIER` | Oversampling factor per path | `4` |
| `OPENMEMORY_SEARCH__VECTOR_WEIGHT` | Vector score weight (0.0–1.0) | `0.7` |
| `OPENMEMORY_SEARCH__TEMPORAL_DECAY_RATE` | Score decay per day of age | `0.0` |
| `OPENMEMORY_SEARCH__MMR_LAMBDA` | MMR diversity (0 = disabled) | `0.0` |

**Chunking**

| Variable | Description | Default |
|---|---|---|
| `OPENMEMORY_CHUNKING__TOKENS` | Target chunk size in tokens | `400` |
| `OPENMEMORY_CHUNKING__OVERLAP` | Overlap between chunks in tokens | `80` |

**Relations**

| Variable | Description | Default |
|---|---|---|
| `OPENMEMORY_RELATIONS__DEDUP_THRESHOLD` | Cosine similarity threshold for dedup | `0.92` |

**Bootstrap**

| Variable | Description | Default |
|---|---|---|
| `OPENMEMORY_BOOTSTRAP__MAX_CHARS_PER_FILE` | Max chars per injected file before truncation | `20000` |
| `OPENMEMORY_BOOTSTRAP__MAX_TOTAL_CHARS` | Max total chars across all injected files | `150000` |
| `OPENMEMORY_BOOTSTRAP__INJECT_LONG_TERM_MEMORY` | Inject MEMORY.md | `true` |
| `OPENMEMORY_BOOTSTRAP__INJECT_USER_PROFILE` | Inject USER.md | `true` |
| `OPENMEMORY_BOOTSTRAP__INJECT_AGENTS` | Inject AGENTS.md | `true` |
| `OPENMEMORY_BOOTSTRAP__INJECT_DAILY_LOGS` | Inject today's + yesterday's daily logs | `true` |
| `OPENMEMORY_BOOTSTRAP__INJECT_RELATIONS` | Inject RELATIONS.md | `true` |
| `OPENMEMORY_BOOTSTRAP__SYNC_RELATIONS_ON_BOOTSTRAP` | Reconcile SQLite relations from RELATIONS.md at session start. Enable when you edit RELATIONS.md manually outside the agent so that changes are reflected at the next session start. | `false` |

**General**

| Variable | Description | Default |
|---|---|---|
| `OPENMEMORY_ROOT_DIR` | Base directory for all workspaces | `~/.openmemory` |
| `OPENMEMORY_WORKSPACE` | Default workspace name | `default` |

**MCP Server**

| Variable | Description | Default |
|---|---|---|
| `OPENMEMORY_MCP_HOST` | Host address the server binds to | `0.0.0.0` |
| `OPENMEMORY_MCP_PORT` | TCP port the server listens on | `4242` |

**Configuration priority (highest wins):**

```
constructor kwargs  >  environment variables  >  $OPENMEMORY_ROOT_DIR/.env / ./.env  >  $OPENMEMORY_ROOT_DIR/openmemory.yaml / ./openmemory.yaml  >  built-in defaults
```
