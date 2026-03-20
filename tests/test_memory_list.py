"""Tests for the memory_list tool."""
from __future__ import annotations


class TestMemoryListFiles:
    def test_list_empty_workspace_returns_ok(self, session):
        r = session.execute_tool("memory_list")
        assert r["status"] == "ok"
        assert "files" in r
        assert isinstance(r["files"], list)

    def test_list_count_matches_files_length(self, session):
        r = session.execute_tool("memory_list")
        assert r["count"] == len(r["files"])

    def test_list_after_long_term_write_shows_memory_md(self, session):
        session.execute_tool("memory_write", content="Something.", tier="long_term")
        r = session.execute_tool("memory_list")
        files = [entry["file"] for entry in r["files"]]
        assert any("MEMORY.md" in f for f in files)

    def test_list_after_relate_shows_relations_md(self, session):
        session.execute_tool(
            "memory_relate", subject="X", predicate="knows", object="Y"
        )
        r = session.execute_tool("memory_list")
        files = [entry["file"] for entry in r["files"]]
        assert any("RELATIONS.md" in f for f in files)

    def test_list_file_entries_have_line_count(self, session):
        session.execute_tool("memory_write", content="Entry 1.", tier="long_term")
        r = session.execute_tool("memory_list")
        for entry in r["files"]:
            assert "line_count" in entry
            assert isinstance(entry["line_count"], int)

    def test_list_target_files_explicit(self, session):
        session.execute_tool("memory_write", content="Explicit target test.", tier="long_term")
        r = session.execute_tool("memory_list", target="files")
        assert r["status"] == "ok"
        assert "files" in r


class TestMemoryListDaily:
    def test_list_daily_empty_returns_ok(self, session):
        r = session.execute_tool("memory_list", target="daily")
        assert r["status"] == "ok"
        assert r["count"] == 0
        assert r["daily_files"] == []

    def test_list_daily_after_write_shows_file(self, session):
        session.execute_tool("memory_write", content="Daily note.", tier="daily")
        r = session.execute_tool("memory_list", target="daily")
        assert r["status"] == "ok"
        assert r["count"] >= 1
        assert len(r["daily_files"]) >= 1

    def test_list_daily_filenames_are_date_formatted(self, session):
        session.execute_tool("memory_write", content="Date format test.", tier="daily")
        r = session.execute_tool("memory_list", target="daily")
        for name in r["daily_files"]:
            # Expect YYYY-MM-DD.md
            parts = name.replace(".md", "").split("-")
            assert len(parts) == 3, f"Unexpected daily filename: {name}"
            assert parts[0].isdigit() and len(parts[0]) == 4


class TestMemoryListFilePeek:
    def test_peek_specific_file(self, session):
        for i in range(5):
            session.execute_tool("memory_write", content=f"Peek line {i}.", tier="long_term")

        r = session.execute_tool("memory_list", file="MEMORY.md")
        assert r["status"] == "ok"
        assert "line_count" in r
        assert "preview" in r
        assert isinstance(r["preview"], list)

    def test_peek_preview_max_20_lines(self, session):
        for i in range(30):
            session.execute_tool("memory_write", content=f"Preview line {i}.", tier="long_term")

        r = session.execute_tool("memory_list", file="MEMORY.md")
        assert len(r["preview"]) <= 20

    def test_peek_nonexistent_file_returns_error(self, session):
        r = session.execute_tool("memory_list", file="NONEXISTENT.md")
        assert r["status"] == "error"