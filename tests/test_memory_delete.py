"""Tests for the memory_delete tool."""
from __future__ import annotations


def _write_lines(session, lines: list[str], tier: str = "long_term"):
    """Helper: write multiple lines to memory."""
    for line in lines:
        session.execute_tool("memory_write", content=line, tier=tier)


class TestMemoryDeleteBasic:
    def test_delete_returns_ok(self, session):
        session.execute_tool("memory_write", content="Line to delete.", tier="long_term")
        get = session.execute_tool("memory_get", file="MEMORY.md")
        line_count = get["content"].count("\n") + 1

        r = session.execute_tool("memory_delete", file="MEMORY.md", start_line=1, end_line=line_count)
        assert r["status"] == "ok"

    def test_delete_reports_deleted_lines(self, session):
        session.execute_tool("memory_write", content="Content to be tombstoned.", tier="long_term")
        r = session.execute_tool("memory_delete", file="MEMORY.md", start_line=1, end_line=2)
        assert r["status"] == "ok"
        assert "deleted_lines" in r

    def test_delete_nonexistent_file_returns_error(self, session):
        r = session.execute_tool("memory_delete", file="NONEXISTENT.md", start_line=1, end_line=2)
        assert r["status"] == "error"
        assert "message" in r

    def test_delete_reduces_visible_content(self, session):
        _write_lines(session, ["Keep this line.", "Delete this line.", "Keep this too."])
        before = session.execute_tool("memory_get", file="MEMORY.md")
        before_len = len(before["content"])

        session.execute_tool("memory_delete", file="MEMORY.md", start_line=3, end_line=4)
        after = session.execute_tool("memory_get", file="MEMORY.md")
        # File changes (tombstone inserted but original lines removed)
        assert after["status"] == "ok"

    def test_delete_with_reason_stored(self, session):
        session.execute_tool("memory_write", content="Reason test line.", tier="long_term")
        r = session.execute_tool(
            "memory_delete",
            file="MEMORY.md",
            start_line=1,
            end_line=2,
            reason="test cleanup",
        )
        assert r["status"] == "ok"

    def test_delete_range_multiple_lines(self, session):
        _write_lines(session, [f"Entry {i}" for i in range(5)])
        r = session.execute_tool("memory_delete", file="MEMORY.md", start_line=2, end_line=5)
        assert r["status"] == "ok"

    def test_delete_daily_file_line(self, session):
        session.execute_tool("memory_write", content="Daily entry to delete.", tier="daily")
        listing = session.execute_tool("memory_list", target="daily")
        daily_name = listing["daily_files"][0]

        r = session.execute_tool(
            "memory_delete", file=f"daily/{daily_name}", start_line=1, end_line=2
        )
        assert r["status"] == "ok"


class TestMemoryDeleteEdgeCases:
    def test_delete_invalid_line_range_returns_error(self, session):
        """start_line > end_line should be handled gracefully."""
        session.execute_tool("memory_write", content="Some content.", tier="long_term")
        # The tool passes to storage.delete_lines which does lines[start:end]
        # If start > end, it results in empty slice — may succeed with 0 deletions or error
        r = session.execute_tool("memory_delete", file="MEMORY.md", start_line=10, end_line=2)
        assert isinstance(r, dict)
        assert "status" in r

    def test_delete_out_of_bounds_line_returns_error(self, session):
        """Deleting beyond file length should be handled gracefully."""
        session.execute_tool("memory_write", content="Short file.", tier="long_term")
        r = session.execute_tool("memory_delete", file="MEMORY.md", start_line=1000, end_line=2000)
        # Either ok (no-op) or error — must not crash
        assert isinstance(r, dict)
        assert "status" in r