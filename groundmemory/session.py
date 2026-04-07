"""MemorySession - central orchestrator for an groundmemory workspace instance."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from groundmemory.config import groundmemoryConfig
from groundmemory.core.workspace import Workspace
from groundmemory.core.index import MemoryIndex
from groundmemory.core.embeddings import make_provider, EmbeddingProvider
from groundmemory.core.sync import sync_workspace
from groundmemory.tools import build_tool_registry

logger = logging.getLogger(__name__)


class MemorySession:
    """
    A MemorySession binds together a workspace, SQLite index, and embedding
    provider into a single object that tool implementations receive as their
    first argument.

    Usage
    -----
    >>> session = MemorySession.create()          # uses ~/.groundmemory/default
    >>> session = MemorySession.create("project") # named workspace
    >>> result  = session.execute_tool("memory_search", query="Alice")
    >>> prompt  = session.bootstrap()             # system-prompt string
    """

    def __init__(
        self,
        workspace: Workspace,
        index: MemoryIndex,
        provider: EmbeddingProvider,
        config: groundmemoryConfig,
    ) -> None:
        self.workspace = workspace
        self.index = index
        self.provider = provider
        self.config = config
        _, self._tool_runners, self._tool_schemas = build_tool_registry(config)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        workspace_name: str = "default",
        config: groundmemoryConfig | None = None,
    ) -> "MemorySession":
        """
        Create (or reopen) a named workspace session.

        Parameters
        ----------
        workspace_name : str
            Logical name for the workspace.  Becomes a sub-directory under
            the configured base path (default ``~/.groundmemory``).
        config : groundmemoryConfig | None
            Supply a pre-built config or let it load from env / defaults.
        """
        if config is None:
            config = groundmemoryConfig()

        workspace = Workspace(config.root_dir / workspace_name)
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

        If ``config.bootstrap.sync_memory_on_bootstrap`` is ``True``,
        all workspace files are re-indexed (via ``sync_workspace``) before
        context is injected.  Enable this option when you edit memory files
        outside the agent (e.g. in a text editor or via git) so that the
        SQLite/vector index is always consistent with disk at session start.

        Returns the formatted string (may be empty if all files are absent).
        """
        if self.config.bootstrap.sync_memory_on_bootstrap:
            try:
                sync_workspace(
                    workspace=self.workspace,
                    index=self.index,
                    provider=self.provider,
                    chunking=self.config.chunking,
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "sync_memory_on_bootstrap failed; continuing without sync",
                    exc_info=True,
                )

        from groundmemory.bootstrap.injector import build_bootstrap_prompt

        return build_bootstrap_prompt(
            self.workspace,
            self.config.bootstrap,
            dispatcher_mode=self.config.dispatcher_mode,
        )

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
        runner = self._tool_runners.get(tool_name)
        if runner is None:
            from groundmemory.tools.base import err

            return err(
                f"Unknown tool '{tool_name}'. "
                f"Available: {list(self._tool_runners.keys())}"
            )

        try:
            return runner(self, **kwargs)  # type: ignore[call-arg]
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool '%s' raised an exception", tool_name)
            from groundmemory.tools.base import err

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