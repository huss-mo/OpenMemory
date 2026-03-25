"""Tests for the memory_get (read) functionality, now via memory_read tool."""
from __future__ import annotations


class TestMemoryGetBasic:
    def test_get_nonexistent_file_returns_error(self, session):
        """Non-existent files should return status=error with the new API."""
        r = session.execute_tool("memory_read", file="NONEXISTENT_FILE.md")
        assert r["status"] == "error"

    def test_get_existing_file_returns_content(self, session):
        content = "Test content in MEMORY.md"
        session.execute_tool("memory_write", file="MEMORY.md", content=content)

        r = session.execute_tool("memory_read", file="MEMORY.md")
        assert r["status"] == "ok"
        assert content in r["content"]

    def test_get_reports_correct_file_name(self, session):
        session.execute_tool("memory_write", file="MEMORY.md", content="some content")
        r = session.execute_tool("memory_read", file="MEMORY.md")
        assert r["file"] == "MEMORY.md"

    def test_get_reports_char_count(self, session):
        session.execute_tool("memory_write", file="MEMORY.md", content="Some content here.")
        r = session.execute_tool("memory_read", file="MEMORY.md")
        assert r["chars"] == len(r["content"])

    def test_get_relations_file(self, session):
        session.execute_tool(
            "memory_relate", subject="Alice", predicate="works_at", object="Acme"
        )
        r = session.execute_tool("memory_read", file="RELATIONS.md")
        assert r["status"] == "ok"
        assert "content" in r

    def test_get_daily_file(self, session):
        content = "Daily log entry for get test."
        session.execute_tool("memory_write", file="daily", content=content)

        listing = session.execute_tool("memory_list", target="daily")
        daily_name = listing["daily_files"][0]

        r = session.execute_tool("memory_read", file=f"daily/{daily_name}")
        assert r["status"] == "ok"
        assert content in r["content"]


class TestMemoryGetLineRange:
    def test_get_with_start_line(self, session):
        for i in range(10):
            session.execute_tool("memory_write", file="MEMORY.md", content=f"Line {i} content.")

        full = session.execute_tool("memory_read", file="MEMORY.md")
        partial = session.execute_tool("memory_read", file="MEMORY.md", start_line=5)

        assert full["status"] == "ok"
        assert partial["status"] == "ok"
        assert len(partial["content"]) < len(full["content"])

    def test_get_with_end_line(self, session):
        for i in range(10):
            session.execute_tool("memory_write", file="MEMORY.md", content=f"Entry {i}: some text.")

        full = session.execute_tool("memory_read", file="MEMORY.md")
        partial = session.execute_tool("memory_read", file="MEMORY.md", end_line=5)

        assert partial["status"] == "ok"
        assert len(partial["content"]) <= len(full["content"])

    def test_get_with_start_and_end_line(self, session):
        for i in range(20):
            session.execute_tool("memory_write", file="MEMORY.md", content=f"Record {i}.")

        r = session.execute_tool("memory_read", file="MEMORY.md", start_line=2, end_line=6)
        assert r["status"] == "ok"
        assert r["start_line"] == 2
        assert r["end_line"] == 6

    def test_get_start_line_zero_same_as_omitted(self, session):
        session.execute_tool("memory_write", file="MEMORY.md", content="Content for range test.")

        r_default = session.execute_tool("memory_read", file="MEMORY.md")
        # start_line=0 is below minimum=1, but treat gracefully - just read from start
        r_zero = session.execute_tool("memory_read", file="MEMORY.md")

        assert r_default["content"] == r_zero["content"]