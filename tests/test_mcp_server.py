"""Unit tests for groundmemory/mcp_server.py.

Strategy
--------
All tests mock the underlying MemorySession so no filesystem or network I/O
occurs. The focus is on:

1. _unwrap() - success and error path
2. _get_session() - lazy initialisation and singleton behaviour
3. Each MCP tool wrapper - correct argument forwarding and return value
4. Error propagation - tool layer errors surface as ValueError
5. Config-gated tools - memory_list and memory_tool registration
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import groundmemory.mcp_server as mcp_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(**extra) -> dict:
    """Build a minimal success envelope as returned by session.execute_tool."""
    return {"ok": True, "status": "ok", **extra}


def _err(msg: str = "something went wrong") -> dict:
    """Build a minimal error envelope matching base.err() output."""
    return {"status": "error", "message": msg}


# ---------------------------------------------------------------------------
# MCPConfig defaults
# ---------------------------------------------------------------------------


class TestMCPConfigDefaults:
    def test_forwarded_allow_ips_default(self):
        from groundmemory.config import MCPConfig

        cfg = MCPConfig()
        assert cfg.forwarded_allow_ips == "127.0.0.1"

    def test_allowed_hosts_default_empty(self):
        from groundmemory.config import MCPConfig

        cfg = MCPConfig()
        assert cfg.allowed_hosts == ""

    def test_host_default_is_localhost(self):
        from groundmemory.config import MCPConfig

        cfg = MCPConfig()
        assert cfg.host == "127.0.0.1"

    def test_expose_memory_list_default_false(self):
        from groundmemory.config import groundmemoryConfig

        cfg = groundmemoryConfig()
        assert cfg.expose_memory_list is False

    def test_dispatcher_mode_default_false(self):
        from groundmemory.config import groundmemoryConfig

        cfg = groundmemoryConfig()
        assert cfg.dispatcher_mode is False


# ---------------------------------------------------------------------------
# _unwrap
# ---------------------------------------------------------------------------


class TestUnwrap:
    def test_success_returns_json_string(self):
        result = _ok(data="hello")
        out = mcp_mod._unwrap(result)
        assert isinstance(out, str)
        assert json.loads(out) == result

    def test_success_all_fields_preserved(self):
        result = _ok(results=[1, 2, 3], count=3)
        parsed = json.loads(mcp_mod._unwrap(result))
        assert parsed["results"] == [1, 2, 3]
        assert parsed["count"] == 3

    def test_error_raises_value_error(self):
        with pytest.raises(ValueError, match="something went wrong"):
            mcp_mod._unwrap(_err("something went wrong"))

    def test_error_without_message_key_raises_unknown(self):
        with pytest.raises(ValueError, match="unknown error"):
            mcp_mod._unwrap({"status": "error"})

    def test_missing_status_raises(self):
        """A dict with no 'status' key (or status != 'ok') is treated as a failure."""
        with pytest.raises(ValueError):
            mcp_mod._unwrap({"ok": True})  # missing 'status' key → status != "ok"


# ---------------------------------------------------------------------------
# _get_session - lazy init and singleton
# ---------------------------------------------------------------------------


class TestGetSession:
    def setup_method(self):
        """Reset module-level _session before each test."""
        mcp_mod._session = None

    def teardown_method(self):
        """Reset again after each test to avoid cross-test pollution."""
        mcp_mod._session = None

    def test_session_is_none_before_first_call(self):
        assert mcp_mod._session is None

    def test_first_call_creates_session(self):
        mock_session = MagicMock()
        with (
            patch("groundmemory.mcp_server.groundmemoryConfig") as MockCfg,
            patch("groundmemory.session.MemorySession") as MockSess,
        ):
            MockCfg.auto.return_value = MagicMock(workspace="test")
            MockSess.create.return_value = mock_session

            result = mcp_mod._get_session()

            assert result is mock_session
            MockSess.create.assert_called_once()

    def test_second_call_returns_same_instance(self):
        mock_session = MagicMock()
        mcp_mod._session = mock_session

        result1 = mcp_mod._get_session()
        result2 = mcp_mod._get_session()

        assert result1 is mock_session
        assert result2 is mock_session

    def test_existing_session_not_recreated(self):
        existing = MagicMock()
        mcp_mod._session = existing

        with patch("groundmemory.mcp_server.groundmemoryConfig") as MockCfg:
            result = mcp_mod._get_session()
            MockCfg.auto.assert_not_called()

        assert result is existing


# ---------------------------------------------------------------------------
# Fixture: patch _get_session for all tool tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_session():
    """Patch _get_session() to return a MagicMock for the duration of a test."""
    session = MagicMock()
    session.execute_tool.return_value = _ok()
    with patch("groundmemory.mcp_server._get_session", return_value=session):
        yield session


# ---------------------------------------------------------------------------
# memory_write
# ---------------------------------------------------------------------------


class TestMcpMemoryWrite:
    def test_write_returns_json(self, mock_session):
        mock_session.execute_tool.return_value = _ok(file="MEMORY.md", chars_written=20)
        out = mcp_mod.memory_write(content="Hello world.", file="MEMORY.md")
        parsed = json.loads(out)
        assert parsed["ok"] is True
        assert parsed["file"] == "MEMORY.md"

    def test_write_default_file_is_memory_md(self, mock_session):
        mcp_mod.memory_write(content="No file specified.")
        args = mock_session.execute_tool.call_args
        assert args.kwargs.get("file") == "MEMORY.md"

    def test_write_with_search_param_forwarded(self, mock_session):
        mock_session.execute_tool.return_value = _ok(mode="replace_text", replaced=True)
        mcp_mod.memory_write(
            file="USER.md",
            search="old text",
            content="new text",
        )
        args = mock_session.execute_tool.call_args
        assert args.kwargs["search"] == "old text"
        assert args.kwargs["content"] == "new text"

    def test_write_with_start_end_line_forwarded(self, mock_session):
        mock_session.execute_tool.return_value = _ok(mode="replace_lines", replaced_lines="2-3")
        mcp_mod.memory_write(
            file="USER.md",
            start_line=2,
            end_line=3,
            content="new content",
        )
        args = mock_session.execute_tool.call_args
        assert args.kwargs["start_line"] == 2
        assert args.kwargs["end_line"] == 3
        assert args.kwargs["content"] == "new content"

    def test_write_delete_mode_empty_content_forwarded(self, mock_session):
        mock_session.execute_tool.return_value = _ok(mode="delete", lines_deleted=1)
        mcp_mod.memory_write(
            file="USER.md",
            start_line=5,
            end_line=5,
            content="",
        )
        args = mock_session.execute_tool.call_args
        assert args.kwargs["content"] == ""
        assert args.kwargs["start_line"] == 5

    def test_write_error_propagates_as_value_error(self, mock_session):
        mock_session.execute_tool.return_value = _err("content is empty")
        with pytest.raises(ValueError, match="content is empty"):
            mcp_mod.memory_write(content="")

    def test_write_with_tags_forwards_tags(self, mock_session):
        mcp_mod.memory_write(content="Tagged.", file="MEMORY.md", tags=["a", "b"])
        args = mock_session.execute_tool.call_args
        assert args.kwargs["tags"] == ["a", "b"]

    def test_write_none_tags_not_forwarded(self, mock_session):
        mcp_mod.memory_write(content="No tags.", file="MEMORY.md")
        args = mock_session.execute_tool.call_args
        assert args.kwargs.get("tags") is None

    def test_write_omits_search_when_none(self, mock_session):
        mcp_mod.memory_write(content="append only", file="MEMORY.md")
        args = mock_session.execute_tool.call_args
        assert "search" not in args.kwargs

    def test_write_omits_start_line_when_none(self, mock_session):
        mcp_mod.memory_write(content="append only", file="MEMORY.md")
        args = mock_session.execute_tool.call_args
        assert "start_line" not in args.kwargs

    def test_write_omits_end_line_when_none(self, mock_session):
        mcp_mod.memory_write(content="append only", file="MEMORY.md")
        args = mock_session.execute_tool.call_args
        assert "end_line" not in args.kwargs


# ---------------------------------------------------------------------------
# memory_read
# ---------------------------------------------------------------------------


class TestMcpMemoryRead:
    def test_read_get_mode_returns_json(self, mock_session):
        mock_session.execute_tool.return_value = _ok(
            mode="get", content="# MEMORY\nsome text", total_lines=2, file="MEMORY.md"
        )
        out = mcp_mod.memory_read(file="MEMORY.md")
        parsed = json.loads(out)
        assert parsed["ok"] is True
        assert parsed["mode"] == "get"

    def test_read_search_mode_returns_json(self, mock_session):
        mock_session.execute_tool.return_value = _ok(mode="search", results=[], count=0)
        out = mcp_mod.memory_read(query="test query")
        parsed = json.loads(out)
        assert parsed["ok"] is True
        assert parsed["mode"] == "search"

    def test_read_forwards_file(self, mock_session):
        mcp_mod.memory_read(file="USER.md")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["file"] == "USER.md"

    def test_read_forwards_query(self, mock_session):
        mcp_mod.memory_read(query="python async")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["query"] == "python async"

    def test_read_omits_top_k_when_none(self, mock_session):
        mcp_mod.memory_read(query="q")
        args = mock_session.execute_tool.call_args
        assert "top_k" not in args.kwargs

    def test_read_includes_top_k_when_provided(self, mock_session):
        mcp_mod.memory_read(query="q", top_k=10)
        args = mock_session.execute_tool.call_args
        assert args.kwargs["top_k"] == 10

    def test_read_forwards_start_end_line(self, mock_session):
        mcp_mod.memory_read(file="MEMORY.md", start_line=5, end_line=20)
        args = mock_session.execute_tool.call_args
        assert args.kwargs["start_line"] == 5
        assert args.kwargs["end_line"] == 20

    def test_read_omits_start_line_when_none(self, mock_session):
        mcp_mod.memory_read(file="MEMORY.md")
        args = mock_session.execute_tool.call_args
        assert "start_line" not in args.kwargs

    def test_read_omits_end_line_when_none(self, mock_session):
        mcp_mod.memory_read(file="MEMORY.md")
        args = mock_session.execute_tool.call_args
        assert "end_line" not in args.kwargs

    def test_read_daily_file_path(self, mock_session):
        mcp_mod.memory_read(file="daily/2025-03-20.md")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["file"] == "daily/2025-03-20.md"

    def test_read_error_propagates(self, mock_session):
        mock_session.execute_tool.return_value = _err("file not found")
        with pytest.raises(ValueError, match="file not found"):
            mcp_mod.memory_read(file="missing.md")

    def test_read_all_optional_params_forwarded(self, mock_session):
        mcp_mod.memory_read(query="q", top_k=5, file="USER.md")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["top_k"] == 5
        assert args.kwargs["file"] == "USER.md"
        assert args.kwargs["query"] == "q"


# ---------------------------------------------------------------------------
# memory_relate
# ---------------------------------------------------------------------------


class TestMcpMemoryRelate:
    def test_relate_returns_json(self, mock_session):
        mock_session.execute_tool.return_value = _ok(triple="Alice → works_at → Acme Corp")
        out = mcp_mod.memory_relate(subject="Alice", predicate="works_at", object="Acme Corp")
        parsed = json.loads(out)
        assert parsed["ok"] is True

    def test_relate_forwards_required_args(self, mock_session):
        mcp_mod.memory_relate(subject="Alice", predicate="works_at", object="Acme Corp")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["subject"] == "Alice"
        assert args.kwargs["predicate"] == "works_at"
        assert args.kwargs["object"] == "Acme Corp"

    def test_relate_default_note_is_empty(self, mock_session):
        mcp_mod.memory_relate(subject="A", predicate="knows", object="B")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["note"] == ""

    def test_relate_default_source_file(self, mock_session):
        mcp_mod.memory_relate(subject="A", predicate="knows", object="B")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["source_file"] == "RELATIONS.md"

    def test_relate_default_confidence(self, mock_session):
        mcp_mod.memory_relate(subject="A", predicate="knows", object="B")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["confidence"] == 1.0

    def test_relate_custom_note_forwarded(self, mock_session):
        mcp_mod.memory_relate(subject="A", predicate="knows", object="B", note="met at conference")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["note"] == "met at conference"

    def test_relate_custom_confidence_forwarded(self, mock_session):
        mcp_mod.memory_relate(subject="A", predicate="knows", object="B", confidence=0.75)
        args = mock_session.execute_tool.call_args
        assert args.kwargs["confidence"] == 0.75

    def test_relate_default_supersedes_is_false(self, mock_session):
        mcp_mod.memory_relate(subject="A", predicate="knows", object="B")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["supersedes"] is False

    def test_relate_supersedes_true_forwarded(self, mock_session):
        mcp_mod.memory_relate(subject="A", predicate="knows", object="B", supersedes=True)
        args = mock_session.execute_tool.call_args
        assert args.kwargs["supersedes"] is True

    def test_relate_error_propagates(self, mock_session):
        mock_session.execute_tool.return_value = _err("dedup rejected triple")
        with pytest.raises(ValueError, match="dedup rejected triple"):
            mcp_mod.memory_relate(subject="A", predicate="p", object="B")


# ---------------------------------------------------------------------------
# memory_bootstrap (tool)
# ---------------------------------------------------------------------------


class TestMcpMemoryBootstrap:
    def test_bootstrap_returns_string(self, mock_session):
        mock_session.bootstrap.return_value = "## Your Memory Context\n\n### Long-Term Memory\n\nsome facts"
        out = mcp_mod.memory_bootstrap()
        assert isinstance(out, str)
        assert "Memory Context" in out

    def test_bootstrap_calls_session_bootstrap(self, mock_session):
        mock_session.bootstrap.return_value = "context"
        mcp_mod.memory_bootstrap()
        mock_session.bootstrap.assert_called_once()

    def test_bootstrap_returns_placeholder_when_empty(self, mock_session):
        mock_session.bootstrap.return_value = ""
        out = mcp_mod.memory_bootstrap()
        assert "No memory context yet" in out

    def test_bootstrap_returns_placeholder_when_none(self, mock_session):
        mock_session.bootstrap.return_value = None
        out = mcp_mod.memory_bootstrap()
        assert "No memory context yet" in out

    def test_bootstrap_full_content_returned_verbatim(self, mock_session):
        content = (
            "<!-- groundmemory bootstrap start -->\n"
            "## Your Memory Context\nfacts\n"
            "<!-- groundmemory bootstrap end -->"
        )
        mock_session.bootstrap.return_value = content
        out = mcp_mod.memory_bootstrap()
        assert out == content


# ---------------------------------------------------------------------------
# memory_bootstrap_prompt (MCP Prompt)
# ---------------------------------------------------------------------------


class TestMcpMemoryBootstrapPrompt:
    def test_prompt_returns_string(self, mock_session):
        mock_session.bootstrap.return_value = "## Your Memory Context\n\nsome facts"
        out = mcp_mod.memory_bootstrap_prompt()
        assert isinstance(out, str)

    def test_prompt_calls_session_bootstrap(self, mock_session):
        mock_session.bootstrap.return_value = "context"
        mcp_mod.memory_bootstrap_prompt()
        mock_session.bootstrap.assert_called_once()

    def test_prompt_returns_placeholder_when_empty(self, mock_session):
        mock_session.bootstrap.return_value = ""
        out = mcp_mod.memory_bootstrap_prompt()
        assert "No memory context yet" in out

    def test_prompt_content_matches_tool_content(self, mock_session):
        """Tool and Prompt must return identical content for the same workspace state."""
        content = "## Your Memory Context\n\nsome facts"
        mock_session.bootstrap.return_value = content
        tool_out = mcp_mod.memory_bootstrap()
        mock_session.bootstrap.return_value = content
        prompt_out = mcp_mod.memory_bootstrap_prompt()
        assert tool_out == prompt_out


# ---------------------------------------------------------------------------
# Tool registration - verify expected tools are wired up to FastMCP
# ---------------------------------------------------------------------------


class TestMcpToolRegistration:
    """Verify that FastMCP has the core tools and prompt registered."""

    def _tool_names(self) -> set[str]:
        """Extract tool names from the FastMCP instance."""
        try:
            import asyncio
            tools = asyncio.get_event_loop().run_until_complete(mcp_mod.mcp.list_tools())
            return {t.name for t in tools}
        except Exception:
            mgr = getattr(mcp_mod.mcp, "_tool_manager", None) or getattr(mcp_mod.mcp, "tool_manager", None)
            if mgr is not None:
                tools = getattr(mgr, "_tools", None) or getattr(mgr, "tools", {})
                return set(tools.keys())
            return set()

    def _prompt_names(self) -> set[str]:
        """Extract prompt names from the FastMCP instance."""
        try:
            import asyncio
            prompts = asyncio.get_event_loop().run_until_complete(mcp_mod.mcp.list_prompts())
            return {p.name for p in prompts}
        except Exception:
            mgr = getattr(mcp_mod.mcp, "_prompt_manager", None) or getattr(mcp_mod.mcp, "prompt_manager", None)
            if mgr is not None:
                prompts = getattr(mgr, "_prompts", None) or getattr(mgr, "prompts", {})
                return set(prompts.keys())
            return set()

    def test_core_tools_registered(self):
        """Core tools (always present regardless of config) are registered."""
        names = self._tool_names()
        expected_core = {
            "memory_bootstrap",
            "memory_read",
            "memory_write",
            "memory_relate",
        }
        assert expected_core.issubset(names), f"Missing core tools: {expected_core - names}"

    def test_bootstrap_prompt_registered(self):
        names = self._prompt_names()
        assert "memory_bootstrap_prompt" in names, f"Prompt not registered. Found: {names}"

    def test_old_tools_not_registered(self):
        """Old split tools must no longer appear in the registry."""
        names = self._tool_names()
        removed = {"memory_search", "memory_get", "memory_delete"}
        present = removed & names
        assert not present, f"Old tools still registered: {present}"


# ---------------------------------------------------------------------------
# Config-gated tool behaviour
# ---------------------------------------------------------------------------


class TestConfigGatedTools:
    """Verify that expose_memory_list and dispatcher_mode gates work."""

    def test_memory_list_absent_by_default(self):
        """memory_list should NOT be in the FastMCP registry with default config."""
        mgr = getattr(mcp_mod.mcp, "_tool_manager", None) or getattr(mcp_mod.mcp, "tool_manager", None)
        if mgr is None:
            pytest.skip("Cannot introspect FastMCP tool manager")
        tools = getattr(mgr, "_tools", None) or getattr(mgr, "tools", {})
        # Default config has expose_memory_list=False, so memory_list should not be registered
        # (unless the test environment loads a custom config - we skip rather than fail hard)
        from groundmemory.config import groundmemoryConfig
        cfg = groundmemoryConfig.auto()
        if not cfg.expose_memory_list:
            assert "memory_list" not in tools, "memory_list registered despite expose_memory_list=False"

    def test_memory_tool_absent_by_default(self):
        """memory_tool dispatcher should NOT be in the registry with default config."""
        mgr = getattr(mcp_mod.mcp, "_tool_manager", None) or getattr(mcp_mod.mcp, "tool_manager", None)
        if mgr is None:
            pytest.skip("Cannot introspect FastMCP tool manager")
        tools = getattr(mgr, "_tools", None) or getattr(mgr, "tools", {})
        from groundmemory.config import groundmemoryConfig
        cfg = groundmemoryConfig.auto()
        if not cfg.dispatcher_mode:
            assert "memory_tool" not in tools, "memory_tool registered despite dispatcher_mode=False"
