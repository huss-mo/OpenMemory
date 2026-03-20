"""MemorySession — central orchestrator for an OpenMemory workspace instance."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from openmemory.config import OpenMemoryConfig
from openmemory.core.workspace import Workspace
from openmemory.core.index import MemoryIndex
from openmemory.core.embeddings import make_provider, EmbeddingProvider
from openmemory.core.sync import sync_workspace
from openmemory.tools import TOOL_RUNNERS

logger = logging.getLogger(__name__)


class MemorySession:
    """
    A MemorySession binds together a workspace, SQLite index, and embedding
    provider into a single object that tool implementations receive as their
    first argument.

    Usage
    -----
    >>> session = MemorySession.create()          # uses ~/.openmemory/default
    >>> session = MemorySession.create("project") # named workspace
    >>> result  = session.execute_tool("memory_search", query="Alice")
    >>> prompt  = session.bootstrap()             # system-prompt string
    """

    def __init__(
        self,
        workspace: Workspace,
        index: MemoryIndex,
        provider: EmbeddingProvider,
        config: OpenMemoryConfig,
    ) -> None:
        self.workspace = workspace
        self.index = index
        self.provider = provider
        self.config = config

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        workspace_name: str = "default",
        config: OpenMemoryConfig | None = None,
    ) -> "MemorySession":
        """
        Create (or reopen) a named workspace session.

        Parameters
        ----------
        workspace_name : str
            Logical name for the workspace.  Becomes a sub-directory under
            the configured base path (default ``~/.openmemory``).
        config : OpenMemoryConfig | None
            Supply a pre-built config or let it load from env / defaults.
        """
        if config is None:
            config = OpenMemoryConfig()

        workspace = Workspace(config.workspace_path / workspace_name)
        index = MemoryIndex(workspace.db_path)
        provider = make_provider(config.embedding)

        session = cls(workspace, index, provider, config)
        return session

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def sync(self, force: bool = False) -> dict[str, int]:
        """
        Walk all memory files and (re-)index any that have changed.

        Returns a summary dict ``{indexed, skipped, removed}``.
        """
        return sync_workspace(
            workspace=self.workspace,
            index=self.index,
            provider=self.provider,
            chunking=self.config.chunking,
            force=force,
        )

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def bootstrap(self) -> str:
        """
        Build the system-prompt injection string containing the agent's
        long-term memory context.

        Returns the formatted string (may be empty if all files are absent).
        """
        from openmemory.bootstrap.injector import build_bootstrap_prompt

        return build_bootstrap_prompt(self.workspace, self.config.bootstrap)

    # ------------------------------------------------------------------
    # Compaction helpers
    # ------------------------------------------------------------------

    def should_compact(self, current_tokens: int, context_window: int) -> bool:
        """Return True when the session is approaching the compaction threshold."""
        from openmemory.bootstrap.compaction import should_flush

        return should_flush(current_tokens, context_window, self.config.compaction)

    def compaction_prompts(self) -> dict[str, str]:
        """Return ``{system, user}`` prompts for the pre-compaction memory flush."""
        from openmemory.bootstrap.compaction import get_compaction_prompts

        return get_compaction_prompts(self.config.compaction)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def execute_tool(self, tool_name: str, **kwargs: Any) -> dict:
        """
        Dispatch a tool call by name.

        Parameters
        ----------
        tool_name : str
            One of the registered tool names (e.g. ``"memory_search"``).
        **kwargs :
            Arguments forwarded verbatim to the tool's ``run`` function.

        Returns
        -------
        dict
            ``{"ok": True, "data": ...}`` or ``{"ok": False, "error": ...}``
        """
        runner = TOOL_RUNNERS.get(tool_name)
        if runner is None:
            from openmemory.tools.base import err

            return err(
                f"Unknown tool '{tool_name}'. "
                f"Available: {list(TOOL_RUNNERS.keys())}"
            )

        try:
            return runner(self, **kwargs)  # type: ignore[call-arg]
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool '%s' raised an exception", tool_name)
            from openmemory.tools.base import err

            return err(f"Tool '{tool_name}' failed: {exc}")

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "MemorySession":
        return self

    def __exit__(self, *_: Any) -> None:
        try:
            self.index.close()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        self.index.close()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MemorySession(workspace={self.workspace.workspace_path}, "
            f"embedding={self.provider.model_id!r})"
        )