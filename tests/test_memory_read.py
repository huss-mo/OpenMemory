"""Tests for the unified memory_read tool.

Dispatch logic:
  - query provided (no file)      → search mode
  - file provided (no query)      → get mode
  - both provided                 → get mode (file takes precedence for slice reads)
  - neither provided              → error
"""
from __future__ import annotations

import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_user(session, content: str) -> None:
    session.workspace.user_file.write_text(content, encoding="utf-8")


def _write_agents(session, content: str) -> None:
    session.workspace.agents_file.write_text(content, encoding="utf-8")


def _append(session, content: str, file: str = "MEMORY.md") -> None:
    """Append content via memory_write (append mode)."""
    session.execute_tool("memory_write", file=file, content=content)


# ===========================================================================
# GET mode: memory_read(file=...)
# ===========================================================================


class TestMemoryReadGet:
    def test_get_returns_ok(self, session):
        _write_user(session, "Hello world.\n")
        r = session.execute_tool("memory_read", file="USER.md")
        assert r["status"] == "ok"

    def test_get_returns_content_key(self, session):
        _write_user(session, "Sample content.\n")
        r = session.execute_tool("memory_read", file="USER.md")
        assert "content" in r

    def test_get_returns_correct_content(self, session):
        _write_user(session, "Alice is a developer.\n")
        r = session.execute_tool("memory_read", file="USER.md")
        assert "Alice is a developer." in r["content"]

    def test_get_memory_md(self, session):
        _append(session, "Long-term fact.")
        r = session.execute_tool("memory_read", file="MEMORY.md")
        assert r["status"] == "ok"
        assert "Long-term fact." in r["content"]

    def test_get_agents_md(self, session):
        _write_agents(session, "Always respond in English.\n")
        r = session.execute_tool("memory_read", file="AGENTS.md")
        assert r["status"] == "ok"
        assert "Always respond in English." in r["content"]

    def test_get_daily_file_by_name(self, session):
        session.execute_tool("memory_write", file="daily", content="Daily entry A.")
        listing = session.execute_tool("memory_list", target="daily")
        daily_name = listing["daily_files"][0]
        r = session.execute_tool("memory_read", file=f"daily/{daily_name}")
        assert r["status"] == "ok"
        assert "Daily entry A." in r["content"]

    def test_get_daily_shorthand(self, session):
        """'daily' shorthand should resolve to today's file."""
        session.execute_tool("memory_write", file="daily", content="Today's note.")
        r = session.execute_tool("memory_read", file="daily")
        assert r["status"] == "ok"
        assert "Today's note." in r["content"]

    def test_get_nonexistent_file_returns_error(self, session):
        r = session.execute_tool("memory_read", file="NONEXISTENT.md")
        assert r["status"] == "error"
        assert "message" in r

    def test_get_returns_mode_get(self, session):
        _write_user(session, "content\n")
        r = session.execute_tool("memory_read", file="USER.md")
        assert r.get("mode") == "get"

    # --- line slicing (1-indexed) ---

    def test_get_with_start_line_slices_content(self, session):
        _write_user(session, "line 1\nline 2\nline 3\nline 4\n")
        r = session.execute_tool("memory_read", file="USER.md", start_line=2)
        assert r["status"] == "ok"
        assert "line 2" in r["content"]
        assert "line 1" not in r["content"]

    def test_get_with_start_and_end_line(self, session):
        _write_user(session, "a\nb\nc\nd\ne\n")
        r = session.execute_tool("memory_read", file="USER.md", start_line=2, end_line=4)
        assert r["status"] == "ok"
        content = r["content"]
        assert "b" in content
        assert "c" in content
        assert "d" in content
        assert "a" not in content
        assert "e" not in content

    def test_get_single_line(self, session):
        _write_user(session, "first\nTARGET\nthird\n")
        r = session.execute_tool("memory_read", file="USER.md", start_line=2, end_line=2)
        assert r["status"] == "ok"
        assert "TARGET" in r["content"]
        assert "first" not in r["content"]
        assert "third" not in r["content"]

    def test_get_full_file_when_no_lines(self, session):
        _write_user(session, "line A\nline B\nline C\n")
        r = session.execute_tool("memory_read", file="USER.md")
        content = r["content"]
        assert "line A" in content
        assert "line B" in content
        assert "line C" in content

    def test_get_line_numbers_are_1_indexed(self, session):
        """start_line=1 should return the very first line."""
        _write_user(session, "FIRST LINE\nsecond\n")
        r = session.execute_tool("memory_read", file="USER.md", start_line=1, end_line=1)
        assert "FIRST LINE" in r["content"]

    def test_get_returns_total_lines(self, session):
        _write_user(session, "a\nb\nc\n")
        r = session.execute_tool("memory_read", file="USER.md")
        assert "total_lines" in r
        assert r["total_lines"] == 3

    def test_get_returns_file_key(self, session):
        _write_user(session, "content\n")
        r = session.execute_tool("memory_read", file="USER.md")
        assert "file" in r
        assert r["file"] == "USER.md"


# ===========================================================================
# SEARCH mode: memory_read(query=...)
# ===========================================================================


class TestMemoryReadSearch:
    def test_search_returns_ok(self, session):
        _append(session, "Python is a programming language.")
        r = session.execute_tool("memory_read", query="Python")
        assert r["status"] == "ok"

    def test_search_returns_results_key(self, session):
        _append(session, "Machine learning uses data.")
        r = session.execute_tool("memory_read", query="machine learning")
        assert "results" in r

    def test_search_returns_count_key(self, session):
        _append(session, "Some fact to index.")
        r = session.execute_tool("memory_read", query="fact")
        assert "count" in r

    def test_search_finds_relevant_content(self, session):
        _append(session, "Alice prefers dark mode in her editor.")
        r = session.execute_tool("memory_read", query="dark mode editor")
        assert r["status"] == "ok"
        # At least one result should mention the indexed content
        assert r["count"] >= 1

    def test_search_returns_mode_search(self, session):
        _append(session, "Some content.")
        r = session.execute_tool("memory_read", query="content")
        assert r.get("mode") == "search"

    def test_search_with_top_k(self, session):
        for i in range(5):
            _append(session, f"Fact number {i} about coding.")
        r = session.execute_tool("memory_read", query="coding", top_k=2)
        assert r["status"] == "ok"
        assert r["count"] <= 2

    def test_search_empty_query_returns_error(self, session):
        r = session.execute_tool("memory_read", query="")
        assert r["status"] == "error"

    def test_search_whitespace_only_query_returns_error(self, session):
        r = session.execute_tool("memory_read", query="   ")
        assert r["status"] == "error"

    def test_search_with_file_filter(self, session):
        _write_user(session, "User-specific detail.\n")
        _append(session, "General long-term fact.")
        r = session.execute_tool("memory_read", query="detail", file="USER.md")
        assert r["status"] == "ok"

    def test_search_no_results_returns_empty_list(self, session):
        _append(session, "Unrelated content about cooking.")
        r = session.execute_tool("memory_read", query="quantum physics superconductor")
        assert r["status"] == "ok"
        assert isinstance(r["results"], list)

    def test_search_results_contain_text_field(self, session):
        _append(session, "The user enjoys hiking in the mountains.")
        r = session.execute_tool("memory_read", query="hiking mountains")
        assert r["status"] == "ok"
        if r["count"] > 0:
            assert "text" in r["results"][0] or "content" in r["results"][0]

    def test_search_results_contain_source_field(self, session):
        _append(session, "Interesting fact for source check.")
        r = session.execute_tool("memory_read", query="source check fact")
        assert r["status"] == "ok"
        if r["count"] > 0:
            result = r["results"][0]
            # Source should reference a filename, not a tier name
            assert "source" in result or "file" in result


# ===========================================================================
# Dispatch edge cases
# ===========================================================================


class TestMemoryReadDispatch:
    def test_neither_query_nor_file_returns_error(self, session):
        r = session.execute_tool("memory_read")
        assert r["status"] == "error"
        assert "message" in r

    def test_file_only_dispatches_to_get(self, session):
        _write_user(session, "dispatch test\n")
        r = session.execute_tool("memory_read", file="USER.md")
        assert r.get("mode") == "get"

    def test_query_only_dispatches_to_search(self, session):
        _append(session, "dispatch search test")
        r = session.execute_tool("memory_read", query="dispatch search")
        assert r.get("mode") == "search"

    def test_schema_has_required_properties(self, session):
        """Verify the tool schema is accessible and has expected fields."""
        from groundmemory.tools.memory_read import SCHEMA
        assert SCHEMA["name"] == "memory_read"
        props = SCHEMA["parameters"]["properties"]
        assert "query" in props
        assert "file" in props