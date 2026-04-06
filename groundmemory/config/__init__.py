"""
groundmemory configuration - driven by Pydantic Settings.

Priority (highest → lowest):
  1. Constructor kwargs (programmatic overrides)
  2. Environment variables  (groundmemory_* prefix)
  3. .env file             ($groundmemory_ROOT_DIR/.env, then ./.env in cwd)
  4. groundmemory.yaml       ($groundmemory_ROOT_DIR/groundmemory.yaml, then ./groundmemory.yaml in cwd)
  5. Built-in defaults

$groundmemory_ROOT_DIR defaults to ~/.groundmemory (pip installs) and is set to /data
in the official Docker image (mounted from ./data on the host).

Config file locations by install method:
  pip install / editable:  ~/.groundmemory/.env  or  ~/.groundmemory/groundmemory.yaml
  Docker:                  ./data/.env          or  ./data/groundmemory.yaml  (host paths)
  dev / cwd override:      ./.env               or  ./groundmemory.yaml
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Helpers: resolve root_dir and build config file search paths
# ---------------------------------------------------------------------------

def _get_root_dir() -> Path:
    """Return the configured root directory without instantiating groundmemoryConfig.

    Reads groundmemory_ROOT_DIR from the environment (same key pydantic-settings
    uses for groundmemoryConfig.root_dir) and falls back to ~/.groundmemory.
    This avoids a circular dependency when building the env_file list that is
    needed *before* groundmemoryConfig can be instantiated.
    """
    raw = os.environ.get("groundmemory_ROOT_DIR")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".groundmemory"


def _env_file_paths() -> tuple[str, str]:
    """Return the ordered .env search paths as strings for pydantic-settings.

    Search order (first match wins at the pydantic-settings level):
      1. $groundmemory_ROOT_DIR/.env  - global user/Docker config
      2. ./.env                     - cwd override (dev / Docker compose injection)
    """
    root = _get_root_dir()
    return (str(root / ".env"), ".env")


def _seed_example_config() -> None:
    """Copy the bundled example config files into root_dir on first run.

    Copies both ``groundmemory.yaml.example`` and ``.env.example`` from the
    ``groundmemory.config`` package into ``$groundmemory_ROOT_DIR/`` the first
    time ``groundmemory-mcp`` starts.  Each file is only written when it does
    not already exist, so subsequent runs are a no-op.

    Uses ``importlib.resources`` so it works for both wheel installs
    (``pip install groundmemory``) and editable installs (``pip install -e .``).
    """
    import importlib.resources as pkg_resources

    root = _get_root_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        return  # Non-fatal - can't create root dir

    dest = root / "groundmemory.yaml.example"
    if not dest.exists():
        try:
            ref = pkg_resources.files("groundmemory.config").joinpath("groundmemory.yaml.example")
            dest.write_bytes(ref.read_bytes())
        except Exception:
            pass  # Non-fatal - missing example file should never crash the server


def _load_yaml_config(filename: str = "groundmemory.yaml") -> dict[str, Any]:
    """Search for *filename* in root_dir then cwd; return parsed dict or {}.

    Search order (first match wins):
      1. $groundmemory_ROOT_DIR/<filename>  - global user config (~/.groundmemory/ or /data/ in Docker)
      2. ./<filename>                     - cwd override (dev mode / project-level)
    """
    root = _get_root_dir()
    candidates = [root / filename, Path.cwd() / filename]
    for p in candidates:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data
    return {}


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class EmbeddingConfig(BaseSettings):
    """Embedding provider configuration.

    Environment variables (prefix: GROUNDMEMORY_EMBEDDING__):
        PROVIDER    - "local" | "openai" | "none"
        LOCAL_MODEL - sentence-transformers model name
        BASE_URL    - OpenAI-compatible endpoint URL
        API_KEY     - API key for the endpoint
        MODEL       - embedding model name
        BATCH_SIZE  - number of texts per embedding call
    """

    model_config = SettingsConfigDict(
        env_prefix="GROUNDMEMORY_EMBEDDING__",
        env_file=_env_file_paths(),
        extra="ignore",
    )

    # "local"  = sentence-transformers  (uv sync --extra local)
    # "openai" = OpenAI-compatible HTTP API
    # "none"   = BM25-only (no vector search, no extra deps)
    provider: Literal["local", "openai", "none"] = "local"

    # sentence-transformers model (provider="local")
    local_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # OpenAI-compatible endpoint (provider="openai")
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: str = "text-embedding-3-small"

    # Batch size for embedding calls
    batch_size: int = 64

    @field_validator("provider", mode="before")
    @classmethod
    def auto_detect_provider(cls, v: str) -> str:
        """Auto-select openai provider if base_url or OPENAI_API_KEY is set."""
        if v == "local":
            if os.environ.get("GROUNDMEMORY_EMBEDDING__BASE_URL") or os.environ.get(
                "OPENAI_API_KEY"
            ):
                return "openai"
        return v


class ChunkingConfig(BaseSettings):
    """Text chunking configuration.

    Environment variables (prefix: GROUNDMEMORY_CHUNKING__):
        TOKENS  - target chunk size in approximate tokens
        OVERLAP - overlap between chunks in approximate tokens
    """

    model_config = SettingsConfigDict(
        env_prefix="GROUNDMEMORY_CHUNKING__",
        env_file=_env_file_paths(),
        extra="ignore",
    )

    tokens: int = 400
    overlap: int = 80


class SearchConfig(BaseSettings):
    """Hybrid search configuration.

    Environment variables (prefix: GROUNDMEMORY_SEARCH__):
        TOP_K               - number of results to return
        CANDIDATE_MULTIPLIER - candidates fetched per path before merging
        VECTOR_WEIGHT       - weight for vector similarity (0.0-1.0)
        TEMPORAL_DECAY_RATE - score decay per day of age (0 = disabled)
        MMR_LAMBDA          - MMR diversity (0 = disabled, 1 = max diversity)
        RERANK_MODEL        - cross-encoder model name for reranking (None = disabled)
    """

    model_config = SettingsConfigDict(
        env_prefix="GROUNDMEMORY_SEARCH__",
        env_file=_env_file_paths(),
        extra="ignore",
    )

    top_k: int = 6
    candidate_multiplier: int = 4
    # keyword weight = 1 - vector_weight
    vector_weight: float = 0.7
    temporal_decay_rate: float = 0.0
    mmr_lambda: float = 0.0
    # Cross-encoder reranking model (None = disabled).
    # Requires: pip install groundmemory[local]
    # Example: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_model: Optional[str] = None


class CompactionConfig(BaseSettings):
    """Pre-compaction flush configuration.

    When the context window is nearly full the adapter injects a flush message
    that tells the agent to call memory_write for anything worth keeping, before
    the LLM provider silently drops or summarises old messages.

    Environment variables (prefix: GROUNDMEMORY_COMPACTION__):
        ENABLED                 - enable/disable compaction hooks
        CONTEXT_WINDOW_TOKENS   - total token capacity of the model being used
        SOFT_THRESHOLD_TOKENS   - flush when token *usage* reaches this count
                                  (tokens consumed from the start of the window,
                                   not tokens remaining at the end)
        RESERVE_FLOOR_TOKENS    - always keep this many tokens free for the model's
                                  reply; sets a hard flush limit at
                                  context_window_tokens - reserve_floor_tokens
        SYSTEM_PROMPT           - system message injected at the flush turn
        USER_PROMPT             - user message injected at the flush turn
    """

    model_config = SettingsConfigDict(
        env_prefix="GROUNDMEMORY_COMPACTION__",
        env_file=_env_file_paths(),
        extra="ignore",
    )

    enabled: bool = True
    # Total token capacity of the model (used to derive the hard flush limit)
    context_window_tokens: int = 128_000
    # Flush when this many tokens have been *consumed* in the context window
    soft_threshold_tokens: int = 64_000
    # Always keep this many tokens free for the model's reply
    reserve_floor_tokens: int = 32_000
    system_prompt: str = (
        "Session nearing compaction. Store durable memories now before context is summarized."
    )
    user_prompt: str = (
        "Review the conversation and write any lasting facts, decisions, or preferences to memory "
        "using memory_write. Reply DONE when finished, or NO_REPLY if nothing needs saving."
    )


class RelationsConfig(BaseSettings):
    """Relation graph configuration.

    Environment variables (prefix: GROUNDMEMORY_RELATIONS__):
        DEDUP_THRESHOLD - cosine similarity above which two triples are
                          considered semantic duplicates (0.0-1.0).
                          Set to 1.0 to disable semantic dedup (exact-match only).
    """

    model_config = SettingsConfigDict(
        env_prefix="GROUNDMEMORY_RELATIONS__",
        env_file=_env_file_paths(),
        extra="ignore",
    )

    dedup_threshold: float = 0.92


class MCPConfig(BaseSettings):
    """MCP server configuration.

    Environment variables (prefix: GROUNDMEMORY_MCP__):
        HOST                - host address the server binds to
        PORT                - TCP port the server listens on
        FORWARDED_ALLOW_IPS - comma-separated IPs trusted to set X-Forwarded-* headers.
                              Use "*" only when a trusted reverse proxy handles all ingress.
        ALLOWED_HOSTS       - comma-separated additional Host header values accepted by the
                              MCP server (DNS rebinding protection). "localhost" and
                              "127.0.0.1" are always allowed. Add your LAN IP or public
                              hostname when accessing the server from another machine.
        API_KEY             - static bearer token required on every request.
                              When unset (default), no authentication is enforced.
                              Set this when exposing the server beyond localhost.
    """

    model_config = SettingsConfigDict(
        env_prefix="GROUNDMEMORY_MCP__",
        env_file=_env_file_paths(),
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 4242
    # Trusted upstream proxy IPs for X-Forwarded-* headers (uvicorn).
    # Default "127.0.0.1" means only a local proxy is trusted.
    # Set to "*" only when a reverse proxy controls all ingress to this server.
    forwarded_allow_ips: str = "127.0.0.1"
    # Extra Host header values accepted by the MCP server (DNS rebinding protection).
    # "localhost" and "127.0.0.1" are always implicitly allowed.
    # Add your LAN IP (e.g. "192.168.1.50:4242") to allow access from other machines.
    # Separate multiple values with commas. Leave empty (default) for local-only access.
    allowed_hosts: str = ""
    # Static bearer token for request authentication.
    # When None (default), no authentication is enforced.
    # When set, every request must include: Authorization: Bearer <api_key>
    api_key: Optional[str] = None


class BootstrapConfig(BaseSettings):
    """Bootstrap injection configuration.

    Environment variables (prefix: GROUNDMEMORY_BOOTSTRAP__):
        MAX_CHARS_PER_FILE       - max chars per injected file before truncation
        MAX_TOTAL_CHARS          - max chars across all injected files
        INJECT_LONG_TERM_MEMORY  - inject MEMORY.md
        INJECT_USER_PROFILE      - inject USER.md
        INJECT_AGENTS            - inject AGENTS.md
        INJECT_DAILY_LOGS        - inject today's/yesterday's daily logs
        INJECT_RELATIONS         - inject RELATIONS.md
        DAILY_LOG_DAYS           - number of daily log files to inject, counting back
                                   from today (1 = today only, 2 = today+yesterday, 0 = none)
        SYNC_MEMORY_ON_BOOTSTRAP - re-index all workspace files that have changed
                                   since the last session before injecting context.
                                   Uses SHA-256 content hashing, so only files
                                   whose content actually changed are re-chunked
                                   and re-embedded. Relations table is reconciled
                                   automatically when RELATIONS.md is among the
                                   changed files.
                                   Disabled by default; enable when you edit
                                   memory files manually between sessions.
    """

    model_config = SettingsConfigDict(
        env_prefix="GROUNDMEMORY_BOOTSTRAP__",
        env_file=_env_file_paths(),
        extra="ignore",
    )

    max_chars_per_file: int = 10_000
    max_total_chars: int = 50_000
    inject_long_term_memory: bool = True
    inject_user_profile: bool = True
    inject_agents: bool = True
    inject_daily_logs: bool = True
    inject_relations: bool = True
    # Number of daily log files to inject, counting back from today.
    # 1 = today only, 2 = today + yesterday, 0 = disabled (same as inject_daily_logs=False).
    daily_log_days: int = 1
    sync_memory_on_bootstrap: bool = False


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class groundmemoryConfig(BaseSettings):
    """Top-level groundmemory configuration.

    Loaded in priority order:
      1. Constructor kwargs
      2. Environment variables  (groundmemory_* prefix, double-underscore nesting)
      3. .env file
      4. groundmemory.yaml        (searched in cwd, then project root)
      5. Defaults

    Environment variables use double-underscore (__) for nesting:
        groundmemory_EMBEDDING__PROVIDER=openai
        groundmemory_EMBEDDING__BASE_URL=http://localhost:11434/v1
        groundmemory_SEARCH__TOP_K=10

    Tool-set flags (apply to both MCP server and Python API):
        GROUNDMEMORY_EXPOSE_MEMORY_LIST=true  - expose memory_list tool
        GROUNDMEMORY_DISPATCHER_MODE=true     - replace all tools with single memory_tool dispatcher

    See groundmemory.yaml.example for the full YAML reference.
    """

    model_config = SettingsConfigDict(
        env_prefix="GROUNDMEMORY_",
        env_file=_env_file_paths(),
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Root directory for all workspaces; defaults to ~/.groundmemory
    root_dir: Path = Field(default_factory=lambda: Path.home() / ".groundmemory")

    # Workspace name (subdirectory under root_dir/<workspace>/<session>)
    workspace: str = "default"

    # Expose the memory_list tool to models (disabled by default to save tokens).
    # Applies to both the MCP server and the Python API.
    # Env var: GROUNDMEMORY_EXPOSE_MEMORY_LIST=true
    expose_memory_list: bool = False

    # Replace all individual tools with a single memory_tool dispatcher
    # (maximum token efficiency - one tool with one-liner descriptions).
    # Applies to both the MCP server and the Python API.
    # Env var: GROUNDMEMORY_DISPATCHER_MODE=true
    dispatcher_mode: bool = False

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    relations: RelationsConfig = Field(default_factory=RelationsConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)

    @property
    def workspace_path(self) -> Path:
        return self.root_dir / self.workspace

    # ------------------------------------------------------------------
    # YAML config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "groundmemoryConfig":
        """Load config from a YAML file, then overlay env vars on top.

        Args:
            path: Explicit path to a YAML file. If None, searches for
                  ``groundmemory.yaml`` in cwd then project root.
        """
        if path is not None:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = _load_yaml_config()

        # Build nested sub-configs from YAML data, then let env vars win.
        # pydantic-settings gives init kwargs HIGHER priority than env vars,
        # so we must NOT pass a YAML field as a kwarg if an env var is set for it.
        expose_memory_list = data.pop("expose_memory_list", None)
        dispatcher_mode = data.pop("dispatcher_mode", None)
        embedding_data = data.pop("embedding", {})
        chunking_data = data.pop("chunking", {})
        search_data = data.pop("search", {})
        relations_data = data.pop("relations", {})
        compaction_data = data.pop("compaction", {})
        bootstrap_data = data.pop("bootstrap", {})
        mcp_data = data.pop("mcp", {})

        def _filter_env_overrides(yaml_dict: dict, env_prefix: str) -> dict:
            """Remove keys from yaml_dict that are already set via environment variables."""
            filtered = {}
            for k, v in yaml_dict.items():
                env_key = f"{env_prefix}{k.upper()}"
                if not os.environ.get(env_key):
                    filtered[k] = v
            return filtered

        # Instantiate sub-configs (env vars override YAML values automatically)
        embedding = EmbeddingConfig(**_filter_env_overrides(embedding_data, "GROUNDMEMORY_EMBEDDING__"))
        chunking = ChunkingConfig(**_filter_env_overrides(chunking_data, "GROUNDMEMORY_CHUNKING__"))
        search = SearchConfig(**_filter_env_overrides(search_data, "GROUNDMEMORY_SEARCH__"))
        relations = RelationsConfig(**_filter_env_overrides(relations_data, "GROUNDMEMORY_RELATIONS__"))
        compaction = CompactionConfig(**_filter_env_overrides(compaction_data, "GROUNDMEMORY_COMPACTION__"))
        bootstrap = BootstrapConfig(**_filter_env_overrides(bootstrap_data, "GROUNDMEMORY_BOOTSTRAP__"))
        mcp = MCPConfig(**_filter_env_overrides(mcp_data, "GROUNDMEMORY_MCP__"))

        extra: dict = {}
        if expose_memory_list is not None and not os.environ.get("GROUNDMEMORY_EXPOSE_MEMORY_LIST"):
            extra["expose_memory_list"] = expose_memory_list
        if dispatcher_mode is not None and not os.environ.get("GROUNDMEMORY_DISPATCHER_MODE"):
            extra["dispatcher_mode"] = dispatcher_mode

        return cls(
            embedding=embedding,
            chunking=chunking,
            search=search,
            relations=relations,
            compaction=compaction,
            bootstrap=bootstrap,
            mcp=mcp,
            **extra,
            **data,
        )

    @classmethod
    def auto(cls) -> "groundmemoryConfig":
        """Auto-load: YAML file if present, otherwise pure env/defaults.

        This is the recommended factory for most use cases - it respects the
        full priority chain without requiring you to specify a file path.
        """
        yaml_data = _load_yaml_config()
        if yaml_data:
            return cls.from_yaml.__func__(cls, None)  # type: ignore[attr-defined]
        return cls()
