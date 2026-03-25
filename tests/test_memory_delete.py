"""Tests for the delete mode of the memory_write tool (hard-delete).

The internal memory_delete helper is still valid but no longer in ALL_TOOLS.
We test the public surface: memory_write(file=..., start_line=N, end_line=M, content="")
which triggers hard-delete (physical line removal, no tombstone).
"""
from __future__ import annotations


def _seed_user(session, *lines: str) -> None:
    """Seed USER.md with direct writes (bypasses dedup for test setup)."""
    ws = session.workspace
    ws.user_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _seed_agents(session, *lines: str) -> None:
    """Seed AGENTS.md with direct writes."""
    ws = session.workspace
    ws.agents_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _get(session, file: str) -> str:
    """Read file content via memory_read."""
    r = session.execute_tool("memory_read", file=file)
    return r.get("content", "")


class TestMemoryDeleteBasic:
    def test_delete_returns_ok(self, session):
        _seed_user(session, "User line A.", "User line B.")
        r = session.execute_tool(
            "memory_write", file="USER.md", start_line=1, end_line=1, content=""
        )
        assert r["status"] == "ok"

    def test_delete_reports_mode_delete(self, session):
        _seed_user(session, "Content to be deleted.", "Second line.")
        r = session.execute_tool(
            "memory_write", file="USER.md", start_line=1, end_line=1, content=""
        )
        assert r["status"] == "ok"
        assert r.get("mode") == "delete"

    def test_delete_nonexistent_file_returns_error(self, session):
        r = session.execute_tool(
            "memory_write", file="NONEXISTENT.md", start_line=1, end_line=2, content=""
        )
        assert r["status"] == "error"
        assert "message" in r

    def test_delete_physically_removes_line_no_tombstone(self, session):
        _seed_user(session, "Keep this.", "Delete this.", "Keep this too.")
        content_before = _get(session, "USER.md")
        assert "Delete this." in content_before

        session.execute_tool(
            "memory_write", file="USER.md", start_line=2, end_line=2, content=""
        )
        content_after = _get(session, "USER.md")

        # Line is physically gone - no tombstone comment
        assert "Delete this." not in content_after
        assert "<!-- deleted" not in content_after
        # Surrounding lines remain
        assert "Keep this." in content_after
        assert "Keep this too." in content_after

    def test_delete_range_multiple_lines(self, session):
        _seed_user(session, "Entry 0", "Entry 1", "Entry 2", "Entry 3", "Entry 4")
        r = session.execute_tool(
            "memory_write", file="USER.md", start_line=2, end_line=4, content=""
        )
        assert r["status"] == "ok"
        content = _get(session, "USER.md")
        assert "Entry 1" not in content
        assert "Entry 2" not in content
        assert "Entry 3" not in content
        assert "Entry 0" in content
        assert "Entry 4" in content

    def test_delete_on_agents_md(self, session):
        _seed_agents(session, "Rule A.", "Rule B.", "Rule C.")
        r = session.execute_tool(
            "memory_write", file="AGENTS.md", start_line=2, end_line=2, content=""
        )
        assert r["status"] == "ok"
        content = _get(session, "AGENTS.md")
        assert "Rule B." not in content
        assert "<!-- deleted" not in content
        assert "Rule A." in content
        assert "Rule C." in content

    def test_delete_first_line(self, session):
        _seed_user(session, "FIRST", "second", "third")
        session.execute_tool(
            "memory_write", file="USER.md", start_line=1, end_line=1, content=""
        )
        content = _get(session, "USER.md")
        assert "FIRST" not in content
        assert "second" in content
        assert "third" in content

    def test_delete_last_line(self, session):
        _seed_user(session, "first", "second", "LAST")
        session.execute_tool(
            "memory_write", file="USER.md", start_line=3, end_line=3, content=""
        )
        content = _get(session, "USER.md")
        assert "LAST" not in content
        assert "first" in content
        assert "second" in content

    def test_delete_all_lines(self, session):
        _seed_user(session, "only", "these", "lines")
        session.execute_tool(
            "memory_write", file="USER.md", start_line=1, end_line=3, content=""
        )
        content = _get(session, "USER.md")
        assert "only" not in content
        assert "these" not in content
        assert "lines" not in content


class TestMemoryDeleteImmutable:
    def test_delete_memory_md_returns_error(self, session):
        """MEMORY.md is append-only; delete should be rejected."""
        r = session.execute_tool(
            "memory_write", file="MEMORY.md", start_line=1, end_line=1, content=""
        )
        assert r["status"] == "error"
        msg = r.get("message", "").lower()
        assert "append-only" in msg or "immutable" in msg

    def test_delete_daily_file_returns_error(self, session):
        session.execute_tool("memory_write", file="daily", content="Daily note.")
        listing = session.execute_tool("memory_list", target="daily")
        daily_name = listing["daily_files"][0]
        r = session.execute_tool(
            "memory_write",
            file=f"daily/{daily_name}",
            start_line=1,
            end_line=1,
            content="",
        )
        assert r["status"] == "error"
        msg = r.get("message", "").lower()
        assert "append-only" in msg or "immutable" in msg


class TestMemoryDeleteEdgeCases:
    def test_delete_invalid_line_range_returns_error(self, session):
        """start_line > end_line should be handled gracefully."""
        _seed_user(session, "Some content.")
        r = session.execute_tool(
            "memory_write", file="USER.md", start_line=10, end_line=2, content=""
        )
        assert isinstance(r, dict)
        assert "status" in r

    def test_delete_out_of_bounds_line_returns_error(self, session):
        """Deleting beyond file length should be handled gracefully."""
        _seed_user(session, "Short file.")
        r = session.execute_tool(
            "memory_write", file="USER.md", start_line=1000, end_line=2000, content=""
        )
        assert isinstance(r, dict)
        assert "status" in r

    def test_delete_reduces_line_count(self, session):
        _seed_user(session, "a", "b", "c", "d", "e")
        before = _get(session, "USER.md").splitlines()
        session.execute_tool(
            "memory_write", file="USER.md", start_line=2, end_line=3, content=""
        )
        after = _get(session, "USER.md").splitlines()
        # 2 lines removed
        assert len(after) == len(before) - 2