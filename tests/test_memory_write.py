"""Tests for the memory_write tool."""
from __future__ import annotations

import pytest


class TestMemoryWriteLongTerm:
    def test_write_returns_ok(self, session):
        r = session.execute_tool("memory_write", content="Alice loves Python.", tier="long_term")
        assert r["status"] == "ok"

    def test_write_long_term_targets_memory_md(self, session):
        r = session.execute_tool("memory_write", content="Bob prefers dark mode.", tier="long_term")
        assert r["file"] == "MEMORY.md"

    def test_write_long_term_content_appears_in_file(self, session):
        content = "Charlie uses vim as his editor."
        session.execute_tool("memory_write", content=content, tier="long_term")

        get = session.execute_tool("memory_get", file="MEMORY.md")
        assert content in get["content"]

    def test_write_returns_chars_written(self, session):
        content = "Test content for char count."
        r = session.execute_tool("memory_write", content=content, tier="long_term")
        assert r["chars_written"] > 0

    def test_write_returns_timestamp(self, session):
        r = session.execute_tool("memory_write", content="Timestamp test.", tier="long_term")
        assert "timestamp" in r

    def test_write_multiple_entries_all_present(self, session):
        items = ["First memory.", "Second memory.", "Third memory."]
        for item in items:
            session.execute_tool("memory_write", content=item, tier="long_term")

        get = session.execute_tool("memory_get", file="MEMORY.md")
        for item in items:
            assert item in get["content"]


class TestMemoryWriteDaily:
    def test_write_daily_returns_ok(self, session):
        r = session.execute_tool("memory_write", content="Daily note.", tier="daily")
        assert r["status"] == "ok"

    def test_write_daily_does_not_write_to_memory_md(self, session):
        session.execute_tool("memory_write", content="Daily note only.", tier="daily")
        get = session.execute_tool("memory_get", file="MEMORY.md")
        assert "Daily note only." not in get["content"]

    def test_write_daily_content_in_daily_file(self, session):
        content = "Today I learned about embeddings."
        session.execute_tool("memory_write", content=content, tier="daily")

        listing = session.execute_tool("memory_list", target="daily")
        assert listing["count"] >= 1

        daily_name = listing["daily_files"][0]
        get = session.execute_tool("memory_get", file=f"daily/{daily_name}")
        assert content in get["content"]

    def test_default_tier_is_daily(self, session):
        """Omitting tier should default to daily."""
        content = "Default tier entry."
        session.execute_tool("memory_write", content=content)

        listing = session.execute_tool("memory_list", target="daily")
        assert listing["count"] >= 1


class TestMemoryWriteWithTags:
    def test_tags_appear_in_file(self, session):
        session.execute_tool(
            "memory_write",
            content="Tagged memory entry.",
            tier="long_term",
            tags=["preference", "ui"],
        )
        get = session.execute_tool("memory_get", file="MEMORY.md")
        assert "#preference" in get["content"]
        assert "#ui" in get["content"]

    def test_tags_without_content_body_still_writes(self, session):
        r = session.execute_tool(
            "memory_write",
            content="Content with single tag.",
            tier="long_term",
            tags=["test"],
        )
        assert r["status"] == "ok"


class TestMemoryWriteValidation:
    def test_empty_content_returns_error(self, session):
        r = session.execute_tool("memory_write", content="")
        assert r["status"] == "error"
        assert "empty" in r["message"].lower()

    def test_whitespace_only_content_returns_error(self, session):
        r = session.execute_tool("memory_write", content="   \n\t  ")
        assert r["status"] == "error"

    def test_unknown_tier_still_handled(self, session):
        """An unknown tier falls back to daily behavior (or errors gracefully)."""
        r = session.execute_tool("memory_write", content="Unknown tier test.", tier="unknown")
        assert isinstance(r, dict)
        assert "status" in r