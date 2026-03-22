"""
OpenMemory configuration — driven by Pydantic Settings.

Priority (highest → lowest):
  1. Constructor kwargs (programmatic overrides)
  2. Environment variables  (OPENMEMORY_* prefix)
  3. .env file             ($OPENMEMORY_ROOT_DIR/.env, then ./.env in cwd)
  4. openmemory.yaml       ($OPENMEMORY_ROOT_DIR/openmemory.yaml, then ./openmemory.yaml in cwd)
  5. Built-in defaults

$OPENMEMORY_ROOT_DIR defaults to ~/.openmemory (pip installs) and is set to /data
in the official Docker image (mounted from ./data on the host).

Config file locations by install method:
  pip install / editable:  ~/.openmemory/.env  or  ~/.openmemory/openmemory.yaml
  Docker:                  ./data/.env          or  ./data/openmemory.yaml  (host paths)
  dev / cwd override:      ./.env               or  ./openmemory.yaml
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
    """Return the configured root directory without instantiating OpenMemoryConfig.

    Reads OPENMEMORY_ROOT_DIR from the environment (same key pydantic-settings
    uses for OpenMemoryConfig.root_dir) and falls back to ~/.openmemory.
    This avoids a circular dependency when building the env_file list that is
    needed *before* OpenMemoryConfig can be instantiated.
    """
    raw = os.environ.get("OPENMEMORY_ROOT_DIR")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".openmemory"


def _env_file_paths() -> tuple[str, str]:
    """Return the ordered .env search paths as strings for pydantic-settings.

    Search order (first match wins at the pydantic-settings level):
      1. $OPENMEMORY_ROOT_DIR/.env  — global user/Docker config
      2. ./.env                     — cwd override (dev / Docker compose injection)
    """
    root = _get_root_dir()
    return (str(root / ".env"), ".env")


def _seed_example_config() -> None:
    """Copy the bundled example config files into root_dir on first run.

    Copies both ``openmemory.yaml.example`` and ``.env.example`` from the
    ``openmemory.config`` package into ``$OPENMEMORY_ROOT_DIR/`` the first
    time ``openmemory-mcp`` starts.  Each file is only written when it does
    not already exist, so subsequent runs are a no-op.

    Uses ``importlib.resources`` so it works for both wheel installs
    (``pip install openmemory``) and editable installs (``pip install -e .``).
    """
    import importlib.resources as pkg_resources

    root = _get_root_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        return  # Non-fatal — can't create root dir

    dest = root / "openmemory.yaml.example"
    if not dest.exists():
        try:
            ref = pkg_resources.files("openmemory.config").joinpath("openmemory.yaml.example")
            dest.write_bytes(ref.read_bytes())
        except Exception:
            pass  # Non-fatal — missing example file should never crash the server


def _load_yaml_config(filename: str = "openmemory.yaml") -> dict[str, Any]:
    """Search for *filename* in root_dir then cwd; return parsed dict or {}.

    Search order (first match wins):
      1. $OPENMEMORY_ROOT_DIR/<filename>  — global user config (~/.openmemory/ or /data/ in Docker)
      2. ./<filename>                     — cwd override (dev mode / project-level)
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

    Environment variables (prefix: OPENMEMORY_EMBEDDING__):
        PROVIDER    - "local" | "openai" | "none"
        LOCAL_MODEL - sentence-transformers model name
        BASE_URL    - OpenAI-compatible endpoint URL
        API_KEY     - API key for the endpoint
        MODEL       - embedding model name
        BATCH_SIZE  - number of texts per embedding call
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENMEMORY_EMBEDDING__",
        env_file=_env_file_paths(),
        extra="ignore",
    )

    # "local"  = sentence-transformers  (uv sync --extra local)
    # "openai" = OpenAI-compatible HTTP API
    # "none"   = BM25-only (no vector search, no extra deps)
    provider: Literal["local", "openai", "none"] = "local"

    # sentence-transformers model (provider="local")
    local_model: str = "all-MiniLM-L6-v2"

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
            if os.environ.get("OPENMEMORY_EMBEDDING__BASE_URL") or os.environ.get(
                "OPENAI_API_KEY"
            ):
                return "openai"
        return v


class ChunkingConfig(BaseSettings):
    """Text chunking configuration.

    Environment variables (prefix: OPENMEMORY_CHUNKING__):
        TOKENS  - target chunk size in approximate tokens
        OVERLAP - overlap between chunks in approximate tokens
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENMEMORY_CHUNKING__",
        env_file=_env_file_paths(),
        extra="ignore",
    )

    tokens: int = 400
    overlap: int = 80


class SearchConfig(BaseSettings):
    """Hybrid search configuration.

    Environment variables (prefix: OPENMEMORY_SEARCH__):
        TOP_K               - number of results to return
        CANDIDATE_MULTIPLIER - candidates fetched per path before merging
        VECTOR_WEIGHT       - weight for vector similarity (0.0-1.0)
        TEMPORAL_DECAY_RATE - score decay per day of age (0 = disabled)
        MMR_LAMBDA          - MMR diversity (0 = disabled, 1 = max diversity)
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENMEMORY_SEARCH__",
        env_file=_env_file_paths(),
        extra="ignore",
    )

    top_k: int = 6
    candidate_multiplier: int = 4
    # keyword weight = 1 - vector_weight
    vector_weight: float = 0.7
    temporal_decay_rate: float = 0.0
    mmr_lambda: float = 0.0


class CompactionConfig(BaseSettings):
    """Pre-compaction flush configuration.

    When the context window is nearly full the adapter injects a flush message
    that tells the agent to call memory_write for anything worth keeping, before
    the LLM provider silently drops or summarises old messages.

    Environment variables (prefix: OPENMEMORY_COMPACTION__):
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
        env_prefix="OPENMEMORY_COMPACTION__",
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

    Environment variables (prefix: OPENMEMORY_RELATIONS__):
        DEDUP_THRESHOLD - cosine similarity above which two triples are
                          considered semantic duplicates (0.0-1.0).
                          Set to 1.0 to disable semantic dedup (exact-match only).
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENMEMORY_RELATIONS__",
        env_file=_env_file_paths(),
        extra="ignore",
    )

    dedup_threshold: float = 0.92


class MCPConfig(BaseSettings):
    """MCP server configuration.

    Environment variables (prefix: OPENMEMORY_MCP__):
        HOST - host address the server binds to
        PORT - TCP port the server listens on
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENMEMORY_MCP__",
        env_file=_env_file_paths(),
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 4242


class BootstrapConfig(BaseSettings):
    """Bootstrap injection configuration.

    Environment variables (prefix: OPENMEMORY_BOOTSTRAP__):
        MAX_CHARS_PER_FILE       - max chars per injected file before truncation
        MAX_TOTAL_CHARS          - max chars across all injected files
        INJECT_LONG_TERM_MEMORY  - inject MEMORY.md
        INJECT_USER_PROFILE      - inject USER.md
        INJECT_AGENTS            - inject AGENTS.md
        INJECT_DAILY_LOGS        - inject today's/yesterday's daily logs
        INJECT_RELATIONS         - inject RELATIONS.md
        SYNC_RELATIONS_ON_BOOTSTRAP - reconcile SQLite relations table from
                                      RELATIONS.md before injecting context.
                                      Useful when a user manually edits
                                      RELATIONS.md outside the agent (e.g. in a
                                      text editor) and wants the internal graph
                                      to be refreshed at the next session start.
                                      Disabled by default; enable it if you edit
                                      RELATIONS.md by hand between sessions.
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENMEMORY_BOOTSTRAP__",
        env_file=_env_file_paths(),
        extra="ignore",
    )

    max_chars_per_file: int = 20_000
    max_total_chars: int = 150_000
    inject_long_term_memory: bool = True
    inject_user_profile: bool = True
    inject_agents: bool = True
    inject_daily_logs: bool = True
    inject_relations: bool = True
    sync_relations_on_bootstrap: bool = False


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class OpenMemoryConfig(BaseSettings):
    """Top-level OpenMemory configuration.

    Loaded in priority order:
      1. Constructor kwargs
      2. Environment variables  (OPENMEMORY_* prefix, double-underscore nesting)
      3. .env file
      4. openmemory.yaml        (searched in cwd, then project root)
      5. Defaults

    Environment variables use double-underscore (__) for nesting:
        OPENMEMORY_EMBEDDING__PROVIDER=openai
        OPENMEMORY_EMBEDDING__BASE_URL=http://localhost:11434/v1
        OPENMEMORY_SEARCH__TOP_K=10

    See openmemory.yaml.example for the full YAML reference.
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENMEMORY_",
        env_file=_env_file_paths(),
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Root directory for all workspaces; defaults to ~/.openmemory
    root_dir: Path = Field(default_factory=lambda: Path.home() / ".openmemory")

    # Workspace name (subdirectory under root_dir/<workspace>/<session>)
    workspace: str = "default"

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
    def from_yaml(cls, path: str | Path | None = None) -> "OpenMemoryConfig":
        """Load config from a YAML file, then overlay env vars on top.

        Args:
            path: Explicit path to a YAML file. If None, searches for
                  ``openmemory.yaml`` in cwd then project root.
        """
        if path is not None:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = _load_yaml_config()

        # Build nested sub-configs from YAML data, then let env vars win.
        # pydantic-settings gives init kwargs HIGHER priority than env vars,
        # so we must NOT pass a YAML field as a kwarg if an env var is set for it.
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
        embedding = EmbeddingConfig(**_filter_env_overrides(embedding_data, "OPENMEMORY_EMBEDDING__"))
        chunking = ChunkingConfig(**_filter_env_overrides(chunking_data, "OPENMEMORY_CHUNKING__"))
        search = SearchConfig(**_filter_env_overrides(search_data, "OPENMEMORY_SEARCH__"))
        relations = RelationsConfig(**_filter_env_overrides(relations_data, "OPENMEMORY_RELATIONS__"))
        compaction = CompactionConfig(**_filter_env_overrides(compaction_data, "OPENMEMORY_COMPACTION__"))
        bootstrap = BootstrapConfig(**_filter_env_overrides(bootstrap_data, "OPENMEMORY_BOOTSTRAP__"))
        mcp = MCPConfig(**_filter_env_overrides(mcp_data, "OPENMEMORY_MCP__"))

        return cls(
            embedding=embedding,
            chunking=chunking,
            search=search,
            relations=relations,
            compaction=compaction,
            bootstrap=bootstrap,
            mcp=mcp,
            **data,
        )

    @classmethod
    def auto(cls) -> "OpenMemoryConfig":
        """Auto-load: YAML file if present, otherwise pure env/defaults.

        This is the recommended factory for most use cases — it respects the
        full priority chain without requiring you to specify a file path.
        """
        yaml_data = _load_yaml_config()
        if yaml_data:
            return cls.from_yaml.__func__(cls, None)  # type: ignore[attr-defined]
        return cls()
