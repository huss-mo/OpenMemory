"""Tests for the memory_get tool."""
from __future__ import annotations


class TestMemoryGetBasic:
    def test_get_nonexistent_file_returns_ok_with_empty(self, session):
        """Non-existent files should return status=ok with empty content (not an error)."""
        r = session.execute_tool("memory_get", file="NONEXISTENT_FILE.md")
        assert r["status"] == "ok"
        assert r["content"] == ""
        assert r["exists"] is False

    def test_get_existing_file_returns_content(self, session):
        content = "Test content in MEMORY.md"
        session.execute_tool("memory_write", content=content, tier="long_term")

        r = session.execute_tool("memory_get", file="MEMORY.md")
        assert r["status"] == "ok"
        assert r["exists"] is True
        assert content in r["content"]

    def test_get_reports_correct_file_name(self, session):
        r = session.execute_tool("memory_get", file="MEMORY.md")
        assert r["file"] == "MEMORY.md"

    def test_get_reports_char_count(self, session):
        session.execute_tool("memory_write", content="Some content here.", tier="long_term")
        r = session.execute_tool("memory_get", file="MEMORY.md")
        assert r["chars"] == len(r["content"])

    def test_get_relations_file(self, session):
        session.execute_tool(
            "memory_relate", subject="Alice", predicate="works_at", object="Acme"
        )
        r = session.execute_tool("memory_get", file="RELATIONS.md")
        assert r["status"] == "ok"
        assert r["exists"] is True

    def test_get_daily_file(self, session):
        content = "Daily log entry for get test."
        session.execute_tool("memory_write", content=content, tier="daily")

        listing = session.execute_tool("memory_list", target="daily")
        daily_name = listing["daily_files"][0]

        r = session.execute_tool("memory_get", file=f"daily/{daily_name}")
        assert r["status"] == "ok"
        assert content in r["content"]


class TestMemoryGetLineRange:
    def test_get_with_start_line(self, session):
        for i in range(10):
            session.execute_tool("memory_write", content=f"Line {i} content.", tier="long_term")

        full = session.execute_tool("memory_get", file="MEMORY.md")
        partial = session.execute_tool("memory_get", file="MEMORY.md", start_line=5)

        assert full["status"] == "ok"
        assert partial["status"] == "ok"
        assert len(partial["content"]) < len(full["content"])

    def test_get_with_end_line(self, session):
        for i in range(10):
            session.execute_tool("memory_write", content=f"Entry {i}: some text.", tier="long_term")

        full = session.execute_tool("memory_get", file="MEMORY.md")
        partial = session.execute_tool("memory_get", file="MEMORY.md", end_line=5)

        assert partial["status"] == "ok"
        assert len(partial["content"]) <= len(full["content"])

    def test_get_with_start_and_end_line(self, session):
        for i in range(20):
            session.execute_tool("memory_write", content=f"Record {i}.", tier="long_term")

        r = session.execute_tool("memory_get", file="MEMORY.md", start_line=2, end_line=6)
        assert r["status"] == "ok"
        assert r["start_line"] == 2
        assert r["end_line"] == 6

    def test_get_start_line_zero_same_as_omitted(self, session):
        session.execute_tool("memory_write", content="Content for range test.", tier="long_term")

        r_default = session.execute_tool("memory_get", file="MEMORY.md")
        r_zero = session.execute_tool("memory_get", file="MEMORY.md", start_line=0)

        assert r_default["content"] == r_zero["content"]