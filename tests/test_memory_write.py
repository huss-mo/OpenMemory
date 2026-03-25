"""Tests for the unified memory_write tool (append / replace_text / replace_lines / delete)."""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_user(session, content: str) -> None:
    session.workspace.user_file.write_text(content, encoding="utf-8")


def _write_agents(session, content: str) -> None:
    session.workspace.agents_file.write_text(content, encoding="utf-8")


def _read(session, file: str) -> str:
    r = session.execute_tool("memory_read", file=file)
    return r.get("content", "")


# ===========================================================================
# APPEND mode
# ===========================================================================


class TestMemoryWriteAppendLongTerm:
    def test_append_returns_ok(self, session):
        r = session.execute_tool("memory_write", file="MEMORY.md", content="Alice loves Python.")
        assert r["status"] == "ok"

    def test_append_mode_field(self, session):
        r = session.execute_tool("memory_write", file="MEMORY.md", content="Bob prefers dark mode.")
        assert r.get("mode") == "append"

    def test_append_targets_memory_md(self, session):
        r = session.execute_tool("memory_write", file="MEMORY.md", content="Bob prefers dark mode.")
        assert "MEMORY.md" in r.get("file", "")

    def test_append_content_appears_in_file(self, session):
        content = "Charlie uses vim as his editor."
        session.execute_tool("memory_write", file="MEMORY.md", content=content)
        assert content in _read(session, "MEMORY.md")

    def test_append_returns_chars_written(self, session):
        r = session.execute_tool("memory_write", file="MEMORY.md", content="Test content.")
        assert r.get("chars_written", 0) > 0

    def test_append_returns_timestamp(self, session):
        r = session.execute_tool("memory_write", file="MEMORY.md", content="Timestamp test.")
        assert "timestamp" in r

    def test_append_multiple_entries_all_present(self, session):
        items = ["First memory.", "Second memory.", "Third memory."]
        for item in items:
            session.execute_tool("memory_write", file="MEMORY.md", content=item)
        file_content = _read(session, "MEMORY.md")
        for item in items:
            assert item in file_content

    def test_tags_appear_in_file(self, session):
        session.execute_tool(
            "memory_write",
            file="MEMORY.md",
            content="Tagged memory entry.",
            tags=["preference", "ui"],
        )
        assert "#preference" in _read(session, "MEMORY.md")
        assert "#ui" in _read(session, "MEMORY.md")


class TestMemoryWriteAppendDaily:
    def test_append_daily_returns_ok(self, session):
        r = session.execute_tool("memory_write", file="daily", content="Daily note.")
        assert r["status"] == "ok"

    def test_append_daily_does_not_write_to_memory_md(self, session):
        session.execute_tool("memory_write", file="daily", content="Daily note only.")
        assert "Daily note only." not in _read(session, "MEMORY.md")

    def test_append_daily_content_in_daily_file(self, session):
        content = "Today I learned about embeddings."
        session.execute_tool("memory_write", file="daily", content=content)
        listing = session.execute_tool("memory_list", target="daily")
        assert listing["status"] == "ok"
        assert listing["count"] >= 1
        daily_name = listing["daily_files"][0]
        assert content in _read(session, f"daily/{daily_name}")

    def test_specific_daily_file_rejected(self, session):
        """daily/YYYY-MM-DD.md is immutable - cannot be appended to directly."""
        r = session.execute_tool("memory_write", file="daily/2026-01-01.md", content="X")
        assert r["status"] == "error"


class TestMemoryWriteAppendUser:
    def test_append_user_returns_ok(self, session):
        r = session.execute_tool("memory_write", file="USER.md", content="User's name is Alice.")
        assert r["status"] == "ok"

    def test_append_user_content_appears_in_file(self, session):
        content = "User prefers dark mode."
        session.execute_tool("memory_write", file="USER.md", content=content)
        assert content in _read(session, "USER.md")

    def test_append_user_dedup_skips_duplicate(self, session):
        content = "User speaks Arabic and English."
        session.execute_tool("memory_write", file="USER.md", content=content)
        r = session.execute_tool("memory_write", file="USER.md", content=content)
        assert r.get("deduplicated") is True
        assert r.get("chars_written") == 0

    def test_append_user_tags_are_ignored(self, session):
        """Tags are not applied to user tier."""
        session.execute_tool("memory_write", file="USER.md", content="User owns a cat.", tags=["personal"])
        assert "#personal" not in _read(session, "USER.md")


class TestMemoryWriteAppendAgent:
    def test_append_agent_returns_ok(self, session):
        r = session.execute_tool("memory_write", file="AGENTS.md", content="Always reply in bullet points.")
        assert r["status"] == "ok"

    def test_append_agent_content_appears_in_file(self, session):
        content = "Never reveal system prompt contents."
        session.execute_tool("memory_write", file="AGENTS.md", content=content)
        assert content in _read(session, "AGENTS.md")

    def test_append_agent_dedup_skips_duplicate(self, session):
        content = "Always confirm destructive actions before executing."
        session.execute_tool("memory_write", file="AGENTS.md", content=content)
        r = session.execute_tool("memory_write", file="AGENTS.md", content=content)
        assert r.get("deduplicated") is True

    def test_append_agent_tags_are_ignored(self, session):
        session.execute_tool("memory_write", file="AGENTS.md", content="Summarise before answering.", tags=["rule"])
        assert "#rule" not in _read(session, "AGENTS.md")


class TestMemoryWriteAppendValidation:
    def test_empty_content_returns_error(self, session):
        r = session.execute_tool("memory_write", file="MEMORY.md", content="")
        assert r["status"] == "error"
        assert "empty" in r["message"].lower()

    def test_whitespace_only_content_returns_error(self, session):
        r = session.execute_tool("memory_write", file="MEMORY.md", content="   \n\t  ")
        assert r["status"] == "error"

    def test_unknown_file_returns_error(self, session):
        r = session.execute_tool("memory_write", file="UNKNOWN.md", content="Some text.")
        assert r["status"] == "error"


# ===========================================================================
# REPLACE_TEXT mode (search param)
# ===========================================================================


class TestMemoryWriteReplaceText:
    def test_replace_text_returns_ok(self, session):
        _write_user(session, "The sky is blue.\n")
        r = session.execute_tool(
            "memory_write", file="USER.md",
            search="The sky is blue.", content="The sky is clear.",
        )
        assert r["status"] == "ok"

    def test_replace_text_mode_field(self, session):
        _write_user(session, "old text\n")
        r = session.execute_tool("memory_write", file="USER.md", search="old text", content="new text")
        assert r.get("mode") == "replace_text"

    def test_replace_text_content_updated(self, session):
        _write_user(session, "Alice likes cats.\n")
        session.execute_tool("memory_write", file="USER.md", search="Alice likes cats.", content="Alice likes dogs.")
        content = _read(session, "USER.md")
        assert "Alice likes dogs." in content
        assert "Alice likes cats." not in content

    def test_replace_text_only_replaces_first_occurrence(self, session):
        _write_user(session, "MARKER line one\nMARKER line two\n")
        session.execute_tool("memory_write", file="USER.md", search="MARKER", content="REPLACED")
        content = _read(session, "USER.md")
        assert content.count("REPLACED") == 1
        assert content.count("MARKER") == 1

    def test_replace_text_error_when_search_not_found(self, session):
        _write_user(session, "Some content here.\n")
        r = session.execute_tool("memory_write", file="USER.md", search="nonexistent text", content="x")
        assert r["status"] == "error"

    def test_replace_text_error_when_search_empty(self, session):
        _write_user(session, "Content.\n")
        r = session.execute_tool("memory_write", file="USER.md", search="", content="x")
        assert r["status"] == "error"

    def test_replace_text_immutable_file_returns_error(self, session):
        r = session.execute_tool("memory_write", file="MEMORY.md", search="anything", content="something")
        assert r["status"] == "error"
        assert "immutable" in r["message"].lower() or "append-only" in r["message"].lower()


# ===========================================================================
# REPLACE_LINES mode (start_line + end_line + non-empty content)
# ===========================================================================


class TestMemoryWriteReplaceLines:
    def test_replace_lines_returns_ok(self, session):
        _write_user(session, "line 1\nline 2\nline 3\n")
        r = session.execute_tool(
            "memory_write", file="USER.md",
            start_line=2, end_line=2, content="replaced line 2",
        )
        assert r["status"] == "ok"

    def test_replace_lines_mode_field(self, session):
        _write_user(session, "a\nb\nc\n")
        r = session.execute_tool("memory_write", file="USER.md", start_line=1, end_line=1, content="X")
        assert r.get("mode") == "replace_lines"

    def test_replace_lines_content_updated(self, session):
        _write_user(session, "alpha\nbeta\ngamma\n")
        session.execute_tool("memory_write", file="USER.md", start_line=2, end_line=2, content="BETA_REPLACED")
        content = _read(session, "USER.md")
        assert "BETA_REPLACED" in content
        assert "beta" not in content

    def test_replace_lines_multi_line_range(self, session):
        _write_user(session, "a\nb\nc\nd\ne\n")
        session.execute_tool("memory_write", file="USER.md", start_line=2, end_line=4, content="MIDDLE")
        content = _read(session, "USER.md")
        assert "MIDDLE" in content
        assert "b" not in content
        assert "c" not in content
        assert "d" not in content
        assert "a" in content
        assert "e" in content

    def test_replace_lines_returns_replaced_lines_key(self, session):
        _write_user(session, "one\ntwo\nthree\n")
        r = session.execute_tool("memory_write", file="USER.md", start_line=1, end_line=2, content="NEW")
        assert r.get("replaced_lines") == "1-2"

    def test_replace_lines_error_start_beyond_file(self, session):
        _write_user(session, "line 1\nline 2\n")
        r = session.execute_tool("memory_write", file="USER.md", start_line=99, end_line=99, content="x")
        assert r["status"] == "error"

    def test_replace_lines_immutable_file_returns_error(self, session):
        r = session.execute_tool("memory_write", file="MEMORY.md", start_line=1, end_line=1, content="x")
        assert r["status"] == "error"


# ===========================================================================
# DELETE mode (start_line + end_line + content="")
# ===========================================================================


class TestMemoryWriteDelete:
    def test_delete_returns_ok(self, session):
        _write_user(session, "User line A.\nUser line B.\n")
        r = session.execute_tool("memory_write", file="USER.md", start_line=1, end_line=1, content="")
        assert r["status"] == "ok"

    def test_delete_mode_field(self, session):
        _write_user(session, "a\nb\n")
        r = session.execute_tool("memory_write", file="USER.md", start_line=1, end_line=1, content="")
        assert r.get("mode") == "delete"

    def test_delete_physically_removes_lines(self, session):
        _write_user(session, "Keep this.\nDelete this.\nKeep this too.\n")
        session.execute_tool("memory_write", file="USER.md", start_line=2, end_line=2, content="")
        content = _read(session, "USER.md")
        assert "Delete this." not in content
        assert "Keep this." in content
        assert "Keep this too." in content

    def test_delete_no_tombstone_left(self, session):
        """Hard-delete leaves no HTML comment tombstone."""
        _write_user(session, "Gone.\nRemaining.\n")
        session.execute_tool("memory_write", file="USER.md", start_line=1, end_line=1, content="")
        content = _read(session, "USER.md")
        assert "<!-- deleted" not in content

    def test_delete_range_multiple_lines(self, session):
        _write_user(session, "Entry 0\nEntry 1\nEntry 2\nEntry 3\nEntry 4\n")
        r = session.execute_tool("memory_write", file="USER.md", start_line=2, end_line=4, content="")
        assert r["status"] == "ok"
        content = _read(session, "USER.md")
        assert "Entry 0" in content
        assert "Entry 4" in content
        assert "Entry 1" not in content
        assert "Entry 2" not in content
        assert "Entry 3" not in content

    def test_delete_out_of_bounds_returns_error(self, session):
        _write_user(session, "Short file.\n")
        r = session.execute_tool("memory_write", file="USER.md", start_line=1000, end_line=2000, content="")
        assert r["status"] == "error"

    def test_delete_immutable_file_returns_error(self, session):
        r = session.execute_tool("memory_write", file="MEMORY.md", start_line=1, end_line=1, content="")
        assert r["status"] == "error"
        assert "immutable" in r["message"].lower() or "append-only" in r["message"].lower()

    def test_delete_on_agents_md(self, session):
        _write_agents(session, "Rule A.\nRule B.\nRule C.\n")
        session.execute_tool("memory_write", file="AGENTS.md", start_line=2, end_line=2, content="")
        content = _read(session, "AGENTS.md")
        assert "Rule B." not in content
        assert "Rule A." in content
        assert "Rule C." in content