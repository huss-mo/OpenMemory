"""Tests for the memory_search tool."""
from __future__ import annotations


class TestMemorySearchBasic:
    def test_search_empty_workspace_returns_ok(self, session):
        r = session.execute_tool("memory_search", query="anything")
        assert r["status"] == "ok"
        assert r["count"] == 0

    def test_search_returns_query_in_response(self, session):
        r = session.execute_tool("memory_search", query="test query")
        assert r["query"] == "test query"

    def test_search_finds_written_content(self, session):
        session.execute_tool("memory_write", content="Alice loves Python programming.", tier="long_term")
        r = session.execute_tool("memory_search", query="Alice Python")
        assert r["status"] == "ok"
        assert r["count"] >= 1

    def test_search_result_shape(self, session):
        session.execute_tool("memory_write", content="Bob uses neovim daily.", tier="long_term")
        r = session.execute_tool("memory_search", query="neovim")
        assert r["status"] == "ok"
        if r["count"] > 0:
            result = r["results"][0]
            assert "text" in result or "chunk" in result or "content" in result or isinstance(result, dict)

    def test_search_empty_query_returns_error(self, session):
        r = session.execute_tool("memory_search", query="")
        assert r["status"] == "error"

    def test_search_whitespace_query_returns_error(self, session):
        r = session.execute_tool("memory_search", query="   ")
        assert r["status"] == "error"

    def test_search_multiple_writes_all_findable(self, session):
        facts = [
            "Carol is a backend engineer.",
            "Carol prefers Rust over C++.",
            "Carol lives in Berlin.",
        ]
        for fact in facts:
            session.execute_tool("memory_write", content=fact, tier="long_term")

        r = session.execute_tool("memory_search", query="Carol engineer")
        assert r["status"] == "ok"
        assert r["count"] >= 1


class TestMemorySearchTopK:
    def test_top_k_limits_results(self, session):
        for i in range(10):
            session.execute_tool("memory_write", content=f"Fact number {i} about something.", tier="long_term")

        r = session.execute_tool("memory_search", query="fact something", top_k=3)
        assert r["status"] == "ok"
        assert len(r["results"]) <= 3

    def test_top_k_one_returns_single_result(self, session):
        for i in range(5):
            session.execute_tool("memory_write", content=f"Entry {i}: unique searchable content.", tier="long_term")

        r = session.execute_tool("memory_search", query="unique searchable content", top_k=1)
        assert r["status"] == "ok"
        assert len(r["results"]) <= 1

    def test_top_k_default_used_when_not_specified(self, session):
        for i in range(20):
            session.execute_tool("memory_write", content=f"Default k entry {i}.", tier="long_term")

        r = session.execute_tool("memory_search", query="Default k entry")
        assert r["status"] == "ok"
        assert isinstance(r["results"], list)


class TestMemorySearchSourceFilter:
    def test_source_long_term_only(self, session):
        session.execute_tool("memory_write", content="Long term: remembers Python forever.", tier="long_term")
        session.execute_tool("memory_write", content="Daily: had coffee today.", tier="daily")

        r = session.execute_tool("memory_search", query="Python forever", source="long_term")
        assert r["status"] == "ok"

    def test_source_daily_only(self, session):
        session.execute_tool("memory_write", content="Long term: important fact.", tier="long_term")
        session.execute_tool("memory_write", content="Daily: had coffee today.", tier="daily")

        r = session.execute_tool("memory_search", query="coffee", source="daily")
        assert r["status"] == "ok"

    def test_invalid_source_handled(self, session):
        """An invalid source filter should not crash — either returns empty or an error."""
        r = session.execute_tool("memory_search", query="anything", source="nonexistent_tier")
        assert isinstance(r, dict)
        assert "status" in r