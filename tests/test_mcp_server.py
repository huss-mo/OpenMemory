"""Unit tests for openmemory/mcp_server.py.

Strategy
--------
All tests mock the underlying MemorySession so no filesystem or network I/O
occurs. The focus is on:

1. _unwrap() — success and error path
2. _get_session() — lazy initialisation and singleton behaviour
3. Each MCP tool wrapper — correct argument forwarding and return value
4. Error propagation — tool layer errors surface as ValueError
"""
from __future__ import annotations

import importlib
import json
from unittest.mock import MagicMock, patch, call

import pytest

import openmemory.mcp_server as mcp_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(**extra) -> dict:
    """Build a minimal success envelope as returned by session.execute_tool."""
    return {"ok": True, "status": "ok", **extra}


def _err(msg: str = "something went wrong") -> dict:
    """Build a minimal error envelope matching base.err() output."""
    return {"status": "error", "message": msg}


def _make_mock_session(**tool_returns):
    """Return a MagicMock MemorySession whose execute_tool side_effect is configurable."""
    session = MagicMock()
    session.execute_tool.return_value = _ok()
    for tool_name, retval in tool_returns.items():
        # We'll use side_effect on a per-call basis when needed
        pass
    return session


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
# _get_session — lazy init and singleton
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
            patch("openmemory.mcp_server.OpenMemoryConfig") as MockCfg,
            patch("openmemory.session.MemorySession") as MockSess,
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

        with patch("openmemory.mcp_server.OpenMemoryConfig") as MockCfg:
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
    with patch("openmemory.mcp_server._get_session", return_value=session):
        yield session


# ---------------------------------------------------------------------------
# memory_write
# ---------------------------------------------------------------------------


class TestMcpMemoryWrite:
    def test_write_long_term_returns_json(self, mock_session):
        mock_session.execute_tool.return_value = _ok(file="MEMORY.md", chars_written=20)
        out = mcp_mod.memory_write(content="Hello world.", tier="long_term")
        parsed = json.loads(out)
        assert parsed["ok"] is True
        assert parsed["file"] == "MEMORY.md"

    def test_write_daily_calls_correct_tier(self, mock_session):
        mock_session.execute_tool.return_value = _ok(file="daily/2025-01-01.md")
        mcp_mod.memory_write(content="Daily entry.", tier="daily")
        mock_session.execute_tool.assert_called_once_with(
            "memory_write", content="Daily entry.", tier="daily", tags=None
        )

    def test_write_default_tier_is_long_term(self, mock_session):
        mcp_mod.memory_write(content="No tier specified.")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["tier"] == "long_term"

    def test_write_with_tags_forwards_tags(self, mock_session):
        mcp_mod.memory_write(content="Tagged.", tier="long_term", tags=["a", "b"])
        args = mock_session.execute_tool.call_args
        assert args.kwargs["tags"] == ["a", "b"]

    def test_write_error_propagates_as_value_error(self, mock_session):
        mock_session.execute_tool.return_value = _err("content is empty")
        with pytest.raises(ValueError, match="content is empty"):
            mcp_mod.memory_write(content="")

    def test_write_none_tags_forwarded(self, mock_session):
        mcp_mod.memory_write(content="No tags.")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["tags"] is None


# ---------------------------------------------------------------------------
# memory_search
# ---------------------------------------------------------------------------


class TestMcpMemorySearch:
    def test_search_returns_json(self, mock_session):
        mock_session.execute_tool.return_value = _ok(results=[], count=0)
        out = mcp_mod.memory_search(query="test query")
        parsed = json.loads(out)
        assert parsed["ok"] is True

    def test_search_forwards_query(self, mock_session):
        mcp_mod.memory_search(query="python async")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["query"] == "python async"

    def test_search_omits_top_k_when_none(self, mock_session):
        mcp_mod.memory_search(query="q")
        args = mock_session.execute_tool.call_args
        assert "top_k" not in args.kwargs

    def test_search_includes_top_k_when_provided(self, mock_session):
        mcp_mod.memory_search(query="q", top_k=10)
        args = mock_session.execute_tool.call_args
        assert args.kwargs["top_k"] == 10

    def test_search_omits_source_when_none(self, mock_session):
        mcp_mod.memory_search(query="q")
        args = mock_session.execute_tool.call_args
        assert "source" not in args.kwargs

    def test_search_includes_source_when_provided(self, mock_session):
        mcp_mod.memory_search(query="q", source="long_term")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["source"] == "long_term"

    def test_search_all_optional_params_forwarded(self, mock_session):
        mcp_mod.memory_search(query="q", top_k=5, source="daily")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["top_k"] == 5
        assert args.kwargs["source"] == "daily"

    def test_search_error_propagates(self, mock_session):
        mock_session.execute_tool.return_value = _err("index not ready")
        with pytest.raises(ValueError, match="index not ready"):
            mcp_mod.memory_search(query="q")


# ---------------------------------------------------------------------------
# memory_get
# ---------------------------------------------------------------------------


class TestMcpMemoryGet:
    def test_get_returns_json(self, mock_session):
        mock_session.execute_tool.return_value = _ok(content="# MEMORY\nsome text")
        out = mcp_mod.memory_get(file="MEMORY.md")
        parsed = json.loads(out)
        assert parsed["ok"] is True

    def test_get_forwards_file(self, mock_session):
        mcp_mod.memory_get(file="MEMORY.md")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["file"] == "MEMORY.md"

    def test_get_default_start_line_is_zero(self, mock_session):
        mcp_mod.memory_get(file="MEMORY.md")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["start_line"] == 0

    def test_get_omits_end_line_when_none(self, mock_session):
        mcp_mod.memory_get(file="MEMORY.md")
        args = mock_session.execute_tool.call_args
        assert "end_line" not in args.kwargs

    def test_get_includes_end_line_when_provided(self, mock_session):
        mcp_mod.memory_get(file="MEMORY.md", start_line=5, end_line=20)
        args = mock_session.execute_tool.call_args
        assert args.kwargs["start_line"] == 5
        assert args.kwargs["end_line"] == 20

    def test_get_daily_file_path(self, mock_session):
        mcp_mod.memory_get(file="daily/2025-03-20.md")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["file"] == "daily/2025-03-20.md"

    def test_get_error_propagates(self, mock_session):
        mock_session.execute_tool.return_value = _err("file not found")
        with pytest.raises(ValueError, match="file not found"):
            mcp_mod.memory_get(file="missing.md")


# ---------------------------------------------------------------------------
# memory_list
# ---------------------------------------------------------------------------


class TestMcpMemoryList:
    def test_list_returns_json(self, mock_session):
        mock_session.execute_tool.return_value = _ok(files=["MEMORY.md", "USER.md"])
        out = mcp_mod.memory_list()
        parsed = json.loads(out)
        assert parsed["ok"] is True

    def test_list_default_target_is_files(self, mock_session):
        mcp_mod.memory_list()
        args = mock_session.execute_tool.call_args
        assert args.kwargs["target"] == "files"

    def test_list_with_target_file_forwarded(self, mock_session):
        mcp_mod.memory_list(target="file", file="MEMORY.md")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["target"] == "file"
        assert args.kwargs["file"] == "MEMORY.md"

    def test_list_omits_file_when_none(self, mock_session):
        mcp_mod.memory_list(target="files")
        args = mock_session.execute_tool.call_args
        assert "file" not in args.kwargs

    def test_list_error_propagates(self, mock_session):
        mock_session.execute_tool.return_value = _err("workspace empty")
        with pytest.raises(ValueError, match="workspace empty"):
            mcp_mod.memory_list()


# ---------------------------------------------------------------------------
# memory_delete
# ---------------------------------------------------------------------------


class TestMcpMemoryDelete:
    def test_delete_returns_json(self, mock_session):
        mock_session.execute_tool.return_value = _ok(lines_deleted=3)
        out = mcp_mod.memory_delete(file="MEMORY.md", start_line=10, end_line=13)
        parsed = json.loads(out)
        assert parsed["ok"] is True

    def test_delete_forwards_all_required_args(self, mock_session):
        mcp_mod.memory_delete(file="MEMORY.md", start_line=2, end_line=5)
        args = mock_session.execute_tool.call_args
        assert args.kwargs["file"] == "MEMORY.md"
        assert args.kwargs["start_line"] == 2
        assert args.kwargs["end_line"] == 5

    def test_delete_default_reason(self, mock_session):
        mcp_mod.memory_delete(file="MEMORY.md", start_line=0, end_line=1)
        args = mock_session.execute_tool.call_args
        assert args.kwargs["reason"] == "deleted by agent"

    def test_delete_custom_reason_forwarded(self, mock_session):
        mcp_mod.memory_delete(file="MEMORY.md", start_line=0, end_line=1, reason="no longer relevant")
        args = mock_session.execute_tool.call_args
        assert args.kwargs["reason"] == "no longer relevant"

    def test_delete_error_propagates(self, mock_session):
        mock_session.execute_tool.return_value = _err("line range out of bounds")
        with pytest.raises(ValueError, match="line range out of bounds"):
            mcp_mod.memory_delete(file="MEMORY.md", start_line=99, end_line=100)


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
        content = "<!-- OpenMemory bootstrap start -->\n## Your Memory Context\nfacts\n<!-- OpenMemory bootstrap end -->"
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
        # Reset call count, return same value
        mock_session.bootstrap.return_value = content
        prompt_out = mcp_mod.memory_bootstrap_prompt()
        assert tool_out == prompt_out


# ---------------------------------------------------------------------------
# Tool name registration (FastMCP wiring)
# ---------------------------------------------------------------------------


class TestMcpToolRegistration:
    """Verify that FastMCP has all 7 tools and 1 prompt registered."""

    def _get_registered_tool_names(self):
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

    def _get_registered_prompt_names(self):
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

    def test_all_seven_tools_registered(self):
        names = self._get_registered_tool_names()
        expected = {
            "memory_bootstrap",
            "memory_write",
            "memory_search",
            "memory_get",
            "memory_list",
            "memory_delete",
            "memory_relate",
        }
        assert expected.issubset(names), f"Missing tools: {expected - names}"

    def test_bootstrap_prompt_registered(self):
        names = self._get_registered_prompt_names()
        assert "memory_bootstrap_prompt" in names, f"Prompt not registered. Found: {names}"
