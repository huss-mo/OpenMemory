"""
OpenMemory configuration — driven by Pydantic Settings.

Priority (highest → lowest):
  1. Constructor kwargs (programmatic overrides)
  2. Environment variables  (OPENMEMORY_* prefix)
  3. .env file             (.env in cwd, then project root)
  4. openmemory.yaml       (openmemory.yaml in cwd, then project root)
  5. Built-in defaults

Example .env:
    OPENMEMORY_EMBEDDING__PROVIDER=openai
    OPENMEMORY_EMBEDDING__BASE_URL=http://localhost:11434/v1
    OPENMEMORY_EMBEDDING__MODEL=nomic-embed-text

Example openmemory.yaml:
    embedding:
      provider: openai
      base_url: http://localhost:11434/v1
      model: nomic-embed-text
    search:
      top_k: 8
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# YAML file loader helper
# ---------------------------------------------------------------------------

def _load_yaml_config(filename: str = "openmemory.yaml") -> dict[str, Any]:
    """Search cwd and git-root for *filename*, return parsed dict or {}."""
    candidates = [Path.cwd() / filename]
    # Also check the directory containing this file (project root when installed in editable mode)
    candidates.append(Path(__file__).parent.parent / filename)
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
        PROVIDER    – "local" | "openai" | "none"
        LOCAL_MODEL – sentence-transformers model name
        BASE_URL    – OpenAI-compatible endpoint URL
        API_KEY     – API key for the endpoint
        MODEL       – embedding model name
        BATCH_SIZE  – number of texts per embedding call
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENMEMORY_EMBEDDING__",
        env_file=".env",
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
        TOKENS  – target chunk size in approximate tokens
        OVERLAP – overlap between chunks in approximate tokens
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENMEMORY_CHUNKING__",
        env_file=".env",
        extra="ignore",
    )

    tokens: int = 400
    overlap: int = 80


class SearchConfig(BaseSettings):
    """Hybrid search configuration.

    Environment variables (prefix: OPENMEMORY_SEARCH__):
        TOP_K               – number of results to return
        CANDIDATE_MULTIPLIER – candidates fetched per path before merging
        VECTOR_WEIGHT       – weight for vector similarity (0.0–1.0)
        TEMPORAL_DECAY_RATE – score decay per day of age (0 = disabled)
        MMR_LAMBDA          – MMR diversity (0 = disabled, 1 = max diversity)
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENMEMORY_SEARCH__",
        env_file=".env",
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

    Environment variables (prefix: OPENMEMORY_COMPACTION__):
        ENABLED                – enable/disable compaction hooks
        SOFT_THRESHOLD_TOKENS  – tokens remaining that trigger flush
        RESERVE_FLOOR_TOKENS   – minimum tokens always kept free
        SYSTEM_PROMPT          – system message injected at flush turn
        USER_PROMPT            – user message injected at flush turn
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENMEMORY_COMPACTION__",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = True
    soft_threshold_tokens: int = 4000
    reserve_floor_tokens: int = 20000
    system_prompt: str = (
        "Session nearing compaction. Store durable memories now before context is summarized."
    )
    user_prompt: str = (
        "Review the conversation and write any lasting facts, decisions, or preferences to memory "
        "using memory_write. Reply DONE when finished, or NO_REPLY if nothing needs saving."
    )


class BootstrapConfig(BaseSettings):
    """Bootstrap injection configuration.

    Environment variables (prefix: OPENMEMORY_BOOTSTRAP__):
        MAX_CHARS_PER_FILE    – max chars per injected file before truncation
        MAX_TOTAL_CHARS       – max chars across all injected files
        INJECT_LONG_TERM_MEMORY – inject MEMORY.md
        INJECT_USER_PROFILE   – inject USER.md
        INJECT_AGENTS         – inject AGENTS.md
        INJECT_DAILY_LOGS     – inject today's/yesterday's daily logs
        INJECT_RELATIONS      – inject RELATIONS.md
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENMEMORY_BOOTSTRAP__",
        env_file=".env",
        extra="ignore",
    )

    max_chars_per_file: int = 20_000
    max_total_chars: int = 150_000
    inject_long_term_memory: bool = True
    inject_user_profile: bool = True
    inject_agents: bool = True
    inject_daily_logs: bool = True
    inject_relations: bool = True


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
        env_file=".env",
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
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)

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

        # Build nested sub-configs from YAML data, then let env vars win
        embedding_data = data.pop("embedding", {})
        chunking_data = data.pop("chunking", {})
        search_data = data.pop("search", {})
        compaction_data = data.pop("compaction", {})
        bootstrap_data = data.pop("bootstrap", {})

        # Instantiate sub-configs (env vars override YAML values automatically)
        embedding = EmbeddingConfig(**embedding_data)
        chunking = ChunkingConfig(**chunking_data)
        search = SearchConfig(**search_data)
        compaction = CompactionConfig(**compaction_data)
        bootstrap = BootstrapConfig(**bootstrap_data)

        return cls(
            embedding=embedding,
            chunking=chunking,
            search=search,
            compaction=compaction,
            bootstrap=bootstrap,
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