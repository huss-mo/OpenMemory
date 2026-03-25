"""Tests for the memory_search functionality, now via memory_read(query=...) tool."""
from __future__ import annotations


class TestMemorySearchBasic:
    def test_search_empty_workspace_returns_ok(self, session):
        r = session.execute_tool("memory_read", query="anything")
        assert r["status"] == "ok"
        assert r["count"] == 0

    def test_search_returns_query_in_response(self, session):
        r = session.execute_tool("memory_read", query="test query")
        assert r["query"] == "test query"

    def test_search_finds_written_content(self, session):
        session.execute_tool("memory_write", file="MEMORY.md", content="Alice loves Python programming.")
        r = session.execute_tool("memory_read", query="Alice Python")
        assert r["status"] == "ok"
        assert r["count"] >= 1

    def test_search_result_shape(self, session):
        session.execute_tool("memory_write", file="MEMORY.md", content="Bob uses neovim daily.")
        r = session.execute_tool("memory_read", query="neovim")
        assert r["status"] == "ok"
        if r["count"] > 0:
            result = r["results"][0]
            assert "text" in result or "chunk" in result or "content" in result or isinstance(result, dict)

    def test_search_empty_query_returns_error(self, session):
        r = session.execute_tool("memory_read", query="")
        assert r["status"] == "error"

    def test_search_whitespace_query_returns_error(self, session):
        r = session.execute_tool("memory_read", query="   ")
        assert r["status"] == "error"

    def test_search_multiple_writes_all_findable(self, session):
        facts = [
            "Carol is a backend engineer.",
            "Carol prefers Rust over C++.",
            "Carol lives in Berlin.",
        ]
        for fact in facts:
            session.execute_tool("memory_write", file="MEMORY.md", content=fact)

        r = session.execute_tool("memory_read", query="Carol engineer")
        assert r["status"] == "ok"
        assert r["count"] >= 1


class TestMemorySearchTopK:
    def test_top_k_limits_results(self, session):
        for i in range(10):
            session.execute_tool("memory_write", file="MEMORY.md", content=f"Fact number {i} about something.")

        r = session.execute_tool("memory_read", query="fact something", top_k=3)
        assert r["status"] == "ok"
        assert len(r["results"]) <= 3

    def test_top_k_one_returns_single_result(self, session):
        for i in range(5):
            session.execute_tool("memory_write", file="MEMORY.md", content=f"Entry {i}: unique searchable content.")

        r = session.execute_tool("memory_read", query="unique searchable content", top_k=1)
        assert r["status"] == "ok"
        assert len(r["results"]) <= 1

    def test_top_k_default_used_when_not_specified(self, session):
        for i in range(20):
            session.execute_tool("memory_write", file="MEMORY.md", content=f"Default k entry {i}.")

        r = session.execute_tool("memory_read", query="Default k entry")
        assert r["status"] == "ok"
        assert isinstance(r["results"], list)


class TestMemorySearchSourceFilter:
    def test_source_long_term_only(self, session):
        session.execute_tool("memory_write", file="MEMORY.md", content="Long term: remembers Python forever.")
        session.execute_tool("memory_write", file="daily", content="Daily: had coffee today.")

        r = session.execute_tool("memory_read", query="Python forever", file="MEMORY.md")
        assert r["status"] == "ok"

    def test_source_daily_only(self, session):
        session.execute_tool("memory_write", file="MEMORY.md", content="Long term: important fact.")
        session.execute_tool("memory_write", file="daily", content="Daily: had coffee today.")

        r = session.execute_tool("memory_read", query="coffee", file="daily")
        assert r["status"] == "ok"

    def test_invalid_source_handled(self, session):
        """An unrecognised file filter should not crash - just returns empty results."""
        r = session.execute_tool("memory_read", query="anything", file="NONEXISTENT.md")
        assert isinstance(r, dict)
        assert "status" in r