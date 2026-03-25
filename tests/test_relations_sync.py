"""Tests for all new relation-sync functionality (5-point plan)."""
from __future__ import annotations

import uuid

import pytest

from groundmemory.config import BootstrapConfig, EmbeddingConfig, groundmemoryConfig, SearchConfig
from groundmemory.core.relations import (
    RELATION_LINE_RE,
    _relation_id,
    add_relation,
    parse_relations_from_file,
    sync_relations_from_file,
)
from groundmemory.session import MemorySession
from groundmemory.tools.memory_replace import _validate_relations_replacement

_EM = "\u2014"  # em dash used in RELATIONS.md format


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(tmp_path, **bootstrap_kwargs) -> MemorySession:
    bootstrap = BootstrapConfig(**bootstrap_kwargs) if bootstrap_kwargs else BootstrapConfig()
    cfg = groundmemoryConfig(
        root_dir=tmp_path,
        workspace="test",
        embedding=EmbeddingConfig(provider="none"),
        search=SearchConfig(),
        bootstrap=bootstrap,
    )
    return MemorySession.create(uuid.uuid4().hex[:8], config=cfg)


def _write_relations(session, content: str) -> None:
    session.workspace.relations_file.write_text(content, encoding="utf-8")


# ===========================================================================
# 1. RELATION_LINE_RE
# ===========================================================================


class TestRelationLineRe:
    VALID = [
        f'- [Alice] --leads--> [Auth Team] (2026-03-20) {_EM} "Sprint"',
        "- [Alice] --leads--> [Auth Team] (2026-03-20)",
        "- [Alice] --leads--> [Auth Team]",
        "  - [Bob] --knows--> [Carol]",
        "- [Subj With Spaces] --predicate--> [Obj With Spaces] (2025-01-01)",
        f'- [A] --b--> [C] (2026-01-01) {_EM} "note"',
        f"- [A] --b--> [C] (2026-01-01) {_EM} note without quotes",
    ]

    INVALID = [
        "",
        "# Heading",
        "Some prose text",
        "- Alice --leads--> Auth Team",
        "- [Alice] -> [Bob]",
        "- [Alice] ---> [Bob]",
        "- [Alice] -- leads --> [Bob]",
        "[Alice] --leads--> [Bob]",
        "- [Alice] --leads--> [Bob] (not-a-date)",
    ]

    @pytest.mark.parametrize("line", VALID)
    def test_valid_line_matches(self, line):
        assert RELATION_LINE_RE.match(line), f"Expected match for: {line!r}"

    @pytest.mark.parametrize("line", INVALID)
    def test_invalid_line_does_not_match(self, line):
        assert not RELATION_LINE_RE.match(line), f"Expected NO match for: {line!r}"

    def test_groups_full(self):
        line = f'- [Alice] --leads--> [Auth Team] (2026-03-20) {_EM} "Note"'
        m = RELATION_LINE_RE.match(line)
        assert m is not None
        subj, pred, obj, date, note = m.groups()
        assert subj == "Alice"
        assert pred == "leads"
        assert obj == "Auth Team"
        assert date == "2026-03-20"
        assert note == "Note"

    def test_groups_minimal(self):
        m = RELATION_LINE_RE.match("- [Bob] --knows--> [Carol]")
        assert m is not None
        subj, pred, obj, date, note = m.groups()
        assert subj == "Bob" and pred == "knows" and obj == "Carol"
        assert date is None and note is None

    def test_groups_date_no_note(self):
        m = RELATION_LINE_RE.match("- [Dave] --manages--> [Team] (2025-06-15)")
        assert m is not None
        assert m.group(4) == "2025-06-15"
        assert m.group(5) is None

    def test_leading_whitespace_matches(self):
        assert RELATION_LINE_RE.match("  - [A] --b--> [C]")


# ===========================================================================
# 2. parse_relations_from_file()
# ===========================================================================


class TestParseRelationsFromFile:
    def test_missing_file_returns_empty(self, tmp_path):
        assert parse_relations_from_file(tmp_path / "RELATIONS.md") == []

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "RELATIONS.md"
        f.write_text("", encoding="utf-8")
        assert parse_relations_from_file(f) == []

    def test_parses_single_line(self, tmp_path):
        f = tmp_path / "RELATIONS.md"
        f.write_text("- [Alice] --leads--> [Auth Team] (2026-03-20)\n", encoding="utf-8")
        result = parse_relations_from_file(f)
        assert len(result) == 1
        assert result[0] == {"subject": "Alice", "predicate": "leads", "object": "Auth Team", "note": None}

    def test_parses_note(self, tmp_path):
        f = tmp_path / "RELATIONS.md"
        f.write_text(f'- [Alice] --leads--> [Team] (2026-03-20) {_EM} "Sprint"\n', encoding="utf-8")
        result = parse_relations_from_file(f)
        assert result[0]["note"] == "Sprint"

    def test_skips_invalid_lines(self, tmp_path):
        f = tmp_path / "RELATIONS.md"
        f.write_text(
            "# Relations\n\n- [Alice] --leads--> [Team]\nProse.\n- [Bob] --knows--> [Carol]\n",
            encoding="utf-8",
        )
        result = parse_relations_from_file(f)
        assert len(result) == 2
        assert {r["subject"] for r in result} == {"Alice", "Bob"}

    def test_parses_multiple_lines(self, tmp_path):
        f = tmp_path / "RELATIONS.md"
        f.write_text("- [A] --b--> [C]\n- [D] --e--> [F]\n- [G] --h--> [I]\n", encoding="utf-8")
        assert len(parse_relations_from_file(f)) == 3

    def test_note_is_none_when_absent(self, tmp_path):
        f = tmp_path / "RELATIONS.md"
        f.write_text("- [X] --y--> [Z] (2025-01-01)\n", encoding="utf-8")
        assert parse_relations_from_file(f)[0]["note"] is None

    def test_all_keys_present(self, tmp_path):
        f = tmp_path / "RELATIONS.md"
        f.write_text("- [A] --b--> [C]\n", encoding="utf-8")
        row = parse_relations_from_file(f)[0]
        assert all(k in row for k in ("subject", "predicate", "object", "note"))

    def test_leading_whitespace_accepted(self, tmp_path):
        f = tmp_path / "RELATIONS.md"
        f.write_text("  - [Alice] --leads--> [Team]\n", encoding="utf-8")
        assert len(parse_relations_from_file(f)) == 1


# ===========================================================================
# 3. sync_relations_from_file()
# ===========================================================================


class TestSyncRelationsFromFile:
    def test_empty_file_clears_sqlite(self, session):
        session.execute_tool("memory_relate", subject="Alice", predicate="works_at", object="Acme")
        assert len(session.index.get_all_relations()) == 1

        session.workspace.relations_file.write_text("", encoding="utf-8")
        result = sync_relations_from_file(session.workspace.relations_file, session.index)

        assert result == {"upserted": 0, "deleted": 1, "total_in_file": 0}
        assert len(session.index.get_all_relations()) == 0

    def test_new_lines_upserted(self, session):
        content = "- [Alice] --leads--> [Auth Team]\n- [Bob] --knows--> [Carol]\n"
        session.workspace.relations_file.write_text(content, encoding="utf-8")
        result = sync_relations_from_file(session.workspace.relations_file, session.index)

        assert result["total_in_file"] == 2
        assert result["upserted"] == 2
        assert len(session.index.get_all_relations()) == 2

    def test_idempotent(self, session):
        session.workspace.relations_file.write_text("- [Alice] --leads--> [Team]\n", encoding="utf-8")
        sync_relations_from_file(session.workspace.relations_file, session.index)
        r = sync_relations_from_file(session.workspace.relations_file, session.index)
        assert r["upserted"] == 1
        assert r["deleted"] == 0
        assert len(session.index.get_all_relations()) == 1

    def test_removed_line_deleted_from_sqlite(self, session):
        session.execute_tool("memory_relate", subject="Alice", predicate="works_at", object="Acme")
        session.execute_tool("memory_relate", subject="Bob", predicate="knows", object="Carol")
        assert len(session.index.get_all_relations()) == 2

        session.workspace.relations_file.write_text("- [Alice] --works_at--> [Acme]\n", encoding="utf-8")
        result = sync_relations_from_file(session.workspace.relations_file, session.index)
        assert result["deleted"] == 1
        assert len(session.index.get_all_relations()) == 1
        assert session.index.get_all_relations()[0]["subject"] == "Alice"

    def test_summary_keys_present(self, session):
        session.workspace.relations_file.write_text("- [A] --b--> [C]\n", encoding="utf-8")
        result = sync_relations_from_file(session.workspace.relations_file, session.index)
        assert all(k in result for k in ("upserted", "deleted", "total_in_file"))

    def test_missing_file_returns_zeros(self, session):
        rf = session.workspace.relations_file
        if rf.exists():
            rf.unlink()
        result = sync_relations_from_file(rf, session.index)
        assert result["total_in_file"] == 0
        assert result["upserted"] == 0

    def test_repeated_sync_no_duplicate_rows(self, session):
        session.workspace.relations_file.write_text("- [A] --b--> [C]\n- [D] --e--> [F]\n", encoding="utf-8")
        for _ in range(3):
            sync_relations_from_file(session.workspace.relations_file, session.index)
        assert len(session.index.get_all_relations()) == 2


# ===========================================================================
# 4. _validate_relations_replacement()
# ===========================================================================


class TestValidateRelationsReplacement:
    def test_valid_single_line(self):
        ok, valid, invalid = _validate_relations_replacement("- [Alice] --leads--> [Team] (2026-01-01)")
        assert ok is True and len(valid) == 1 and invalid == []

    def test_valid_multiple_lines(self):
        text = "- [Alice] --leads--> [Team] (2026-01-01)\n- [Bob] --knows--> [Carol] (2026-01-02)\n"
        ok, valid, invalid = _validate_relations_replacement(text)
        assert ok is True and len(valid) == 2

    def test_invalid_single_line(self):
        ok, valid, invalid = _validate_relations_replacement("Alice leads the team")
        assert ok is False and len(invalid) == 1

    def test_blank_lines_skipped(self):
        ok, valid, invalid = _validate_relations_replacement("\n\n- [A] --b--> [C]\n\n")
        assert ok is True

    def test_comment_lines_skipped(self):
        ok, valid, invalid = _validate_relations_replacement(
            "# heading\n- [Alice] --works_at--> [Acme]\n<!-- comment -->\n"
        )
        assert ok is True

    def test_mixed_returns_false(self):
        ok, valid, invalid = _validate_relations_replacement(
            "- [Alice] --leads--> [Team]\nThis line is invalid\n"
        )
        assert ok is False
        assert len(valid) == 1 and len(invalid) == 1
        assert "This line is invalid" in invalid

    def test_empty_string(self):
        ok, valid, invalid = _validate_relations_replacement("")
        assert ok is True and valid == [] and invalid == []

    def test_only_comments_and_blanks(self):
        ok, valid, invalid = _validate_relations_replacement("\n# heading\n\n<!-- comment -->\n")
        assert ok is True

    def test_multiple_invalid_counted(self):
        ok, valid, invalid = _validate_relations_replacement(
            "bad one\nbad two\n- [Good] --relation--> [Fine]\n"
        )
        assert ok is False and len(invalid) == 2 and len(valid) == 1

    def test_returns_three_tuple_of_correct_types(self):
        ok, valid, invalid = _validate_relations_replacement("- [A] --b--> [C]")
        assert isinstance(ok, bool) and isinstance(valid, list) and isinstance(invalid, list)


# ===========================================================================
# 5. memory_replace_text on RELATIONS.md
# ===========================================================================


class TestMemoryReplaceTextRelations:
    _SEARCH = "- [Alice] --leads--> [Auth Team] (2026-03-20)"
    _VALID_REPL = "- [Alice] --leads--> [Auth Team] (2026-04-01)"
    _INVALID_REPL = "Alice just leads the team now"

    def _seed(self, session) -> None:
        _write_relations(
            session,
            "- [Alice] --leads--> [Auth Team] (2026-03-20)\n- [Bob] --knows--> [Carol] (2026-03-20)\n",
        )
        sync_relations_from_file(session.workspace.relations_file, session.index)

    def test_valid_returns_ok(self, session):
        self._seed(session)
        r = session.execute_tool(
            "memory_write", file="RELATIONS.md", search=self._SEARCH, content=self._VALID_REPL
        )
        assert r["status"] == "ok"

    def test_valid_format_confirmed(self, session):
        self._seed(session)
        r = session.execute_tool(
            "memory_write", file="RELATIONS.md", search=self._SEARCH, content=self._VALID_REPL
        )
        assert r.get("relations_format") == "confirmed"

    def test_valid_format_reminder_present(self, session):
        self._seed(session)
        r = session.execute_tool(
            "memory_write", file="RELATIONS.md", search=self._SEARCH, content=self._VALID_REPL
        )
        assert "format_reminder" in r

    def test_valid_relations_synced_present(self, session):
        self._seed(session)
        r = session.execute_tool(
            "memory_write", file="RELATIONS.md", search=self._SEARCH, content=self._VALID_REPL
        )
        assert "relations_synced" in r

    def test_invalid_returns_error(self, session):
        self._seed(session)
        r = session.execute_tool(
            "memory_write", file="RELATIONS.md", search=self._SEARCH, content=self._INVALID_REPL
        )
        assert r["status"] == "error"
        assert "format" in r["message"].lower()

    def test_invalid_does_not_modify_file(self, session):
        self._seed(session)
        original = session.workspace.relations_file.read_text(encoding="utf-8")
        session.execute_tool(
            "memory_write", file="RELATIONS.md", search=self._SEARCH, content=self._INVALID_REPL
        )
        assert session.workspace.relations_file.read_text(encoding="utf-8") == original

    def test_valid_file_content_updated(self, session):
        self._seed(session)
        session.execute_tool(
            "memory_write", file="RELATIONS.md", search=self._SEARCH, content=self._VALID_REPL
        )
        content = session.workspace.relations_file.read_text(encoding="utf-8")
        assert "2026-04-01" in content
        assert "2026-03-20" not in content or content.count("2026-03-20") == 1  # only Bob's line remains


# ===========================================================================
# 6. memory_replace_lines on RELATIONS.md
# ===========================================================================


class TestMemoryReplaceLinesRelations:
    _VALID_LINE = "- [Alice] --leads--> [Auth Team] (2026-04-01)"
    _INVALID_LINE = "Alice just leads the team now"

    def _seed(self, session) -> None:
        _write_relations(
            session,
            "- [Alice] --leads--> [Auth Team] (2026-03-20)\n- [Bob] --knows--> [Carol] (2026-03-20)\n",
        )
        sync_relations_from_file(session.workspace.relations_file, session.index)

    def test_valid_returns_ok(self, session):
        self._seed(session)
        r = session.execute_tool(
            "memory_write", file="RELATIONS.md", start_line=1, end_line=1, content=self._VALID_LINE
        )
        assert r["status"] == "ok"

    def test_valid_format_confirmed(self, session):
        self._seed(session)
        r = session.execute_tool(
            "memory_write", file="RELATIONS.md", start_line=1, end_line=1, content=self._VALID_LINE
        )
        assert r.get("relations_format") == "confirmed"

    def test_valid_format_reminder_present(self, session):
        self._seed(session)
        r = session.execute_tool(
            "memory_write", file="RELATIONS.md", start_line=1, end_line=1, content=self._VALID_LINE
        )
        assert "format_reminder" in r

    def test_valid_relations_synced_present(self, session):
        self._seed(session)
        r = session.execute_tool(
            "memory_write", file="RELATIONS.md", start_line=1, end_line=1, content=self._VALID_LINE
        )
        assert "relations_synced" in r

    def test_invalid_returns_error(self, session):
        self._seed(session)
        r = session.execute_tool(
            "memory_write", file="RELATIONS.md", start_line=1, end_line=1, content=self._INVALID_LINE
        )
        assert r["status"] == "error"
        assert "format" in r["message"].lower()

    def test_invalid_does_not_modify_file(self, session):
        self._seed(session)
        original = session.workspace.relations_file.read_text(encoding="utf-8")
        session.execute_tool(
            "memory_write", file="RELATIONS.md", start_line=1, end_line=1, content=self._INVALID_LINE
        )
        assert session.workspace.relations_file.read_text(encoding="utf-8") == original

    def test_valid_file_content_updated(self, session):
        self._seed(session)
        session.execute_tool(
            "memory_write", file="RELATIONS.md", start_line=1, end_line=1, content=self._VALID_LINE
        )
        content = session.workspace.relations_file.read_text(encoding="utf-8")
        assert "2026-04-01" in content


# ===========================================================================
# 7. memory_delete on RELATIONS.md
# ===========================================================================


class TestMemoryDeleteRelations:
    def _seed(self, session) -> None:
        _write_relations(
            session,
            "- [Alice] --leads--> [Auth Team] (2026-03-20)\n"
            "- [Bob] --knows--> [Carol] (2026-03-20)\n",
        )
        sync_relations_from_file(session.workspace.relations_file, session.index)

    def test_delete_returns_ok(self, session):
        self._seed(session)
        r = session.execute_tool("memory_write", file="RELATIONS.md", start_line=1, end_line=1, content="")
        assert r["status"] == "ok"

    def test_delete_removes_sqlite_row(self, session):
        self._seed(session)
        assert len(session.index.get_all_relations()) == 2

        session.execute_tool("memory_write", file="RELATIONS.md", start_line=1, end_line=1, content="")
        rows = session.index.get_all_relations()
        assert len(rows) == 1
        assert rows[0]["subject"] == "Bob"

    def test_delete_returns_relations_deleted_list(self, session):
        self._seed(session)
        r = session.execute_tool("memory_write", file="RELATIONS.md", start_line=1, end_line=1, content="")
        assert "relations_deleted" in r
        assert len(r["relations_deleted"]) == 1
        assert "Alice" in r["relations_deleted"][0]

    def test_delete_both_lines_removes_both_rows(self, session):
        self._seed(session)
        session.execute_tool("memory_write", file="RELATIONS.md", start_line=1, end_line=2, content="")
        assert len(session.index.get_all_relations()) == 0

    def test_delete_non_relation_line_no_relations_deleted_key(self, session):
        """Deleting a header/blank line should not add relations_deleted key."""
        _write_relations(session, "# Relations\n- [Alice] --leads--> [Team]\n")
        sync_relations_from_file(session.workspace.relations_file, session.index)
        r = session.execute_tool("memory_write", file="RELATIONS.md", start_line=1, end_line=1, content="")
        assert r["status"] == "ok"
        # No relation triple in the deleted slice, so relations_deleted should be absent or empty
        assert r.get("relations_deleted", []) == []

    def test_delete_leaves_no_tombstone(self, session):
        """Hard-delete leaves no HTML comment tombstone."""
        self._seed(session)
        session.execute_tool("memory_write", file="RELATIONS.md", start_line=1, end_line=1, content="")
        content = session.workspace.relations_file.read_text(encoding="utf-8")
        assert "<!-- deleted" not in content

    def test_delete_preserves_other_lines(self, session):
        self._seed(session)
        session.execute_tool("memory_write", file="RELATIONS.md", start_line=1, end_line=1, content="")
        content = session.workspace.relations_file.read_text(encoding="utf-8")
        assert "Bob" in content


# ===========================================================================
# 8. session.bootstrap() with sync_relations_on_bootstrap=True
# ===========================================================================


class TestBootstrapSyncRelations:
    def test_bootstrap_sync_false_by_default(self, tmp_path):
        """BootstrapConfig.sync_relations_on_bootstrap defaults to False."""
        cfg = BootstrapConfig()
        assert cfg.sync_relations_on_bootstrap is False

    def test_bootstrap_sync_can_be_set_true(self, tmp_path):
        cfg = BootstrapConfig(sync_relations_on_bootstrap=True)
        assert cfg.sync_relations_on_bootstrap is True

    def test_bootstrap_with_sync_true_populates_sqlite(self, tmp_path):
        """
        When sync_relations_on_bootstrap=True, bootstrap() reconciles SQLite
        from whatever is in RELATIONS.md at session open time.
        """
        # Step 1: create session, write RELATIONS.md manually (no add_relation)
        s = _make_session(tmp_path, sync_relations_on_bootstrap=True)
        try:
            _write_relations(s, "- [Zara] --owns--> [Laptop]\n")
            # SQLite is empty at this point - the line was written directly
            assert len(s.index.get_all_relations()) == 0

            # Step 2: bootstrap() should trigger the sync
            s.bootstrap()

            assert len(s.index.get_all_relations()) == 1
            assert s.index.get_all_relations()[0]["subject"] == "Zara"
        finally:
            s.close()

    def test_bootstrap_with_sync_false_does_not_populate_sqlite(self, tmp_path):
        """
        When sync_relations_on_bootstrap=False (default), bootstrap() must NOT
        reconcile SQLite - manually written RELATIONS.md lines stay out of SQLite.
        """
        s = _make_session(tmp_path, sync_relations_on_bootstrap=False)
        try:
            _write_relations(s, "- [Zara] --owns--> [Laptop]\n")
            assert len(s.index.get_all_relations()) == 0
            s.bootstrap()
            assert len(s.index.get_all_relations()) == 0
        finally:
            s.close()

    def test_bootstrap_returns_string_with_sync_true(self, tmp_path):
        s = _make_session(tmp_path, sync_relations_on_bootstrap=True)
        try:
            result = s.bootstrap()
            assert isinstance(result, str)
        finally:
            s.close()

    def test_bootstrap_sync_tolerates_missing_relations_file(self, tmp_path):
        """sync_relations_on_bootstrap=True must not raise when RELATIONS.md is absent."""
        s = _make_session(tmp_path, sync_relations_on_bootstrap=True)
        try:
            rf = s.workspace.relations_file
            if rf.exists():
                rf.unlink()
            result = s.bootstrap()  # must not raise
            assert isinstance(result, str)
        finally:
            s.close()


# ===========================================================================
# 9. sync_workspace() / sync_file() trigger relation sync
# ===========================================================================


class TestSyncTriggerRelations:
    def test_sync_file_syncs_relations(self, session):
        """sync_file on RELATIONS.md must reconcile the SQLite relations table."""
        from groundmemory.core.sync import sync_file

        _write_relations(session, "- [Alice] --leads--> [Team]\n")
        assert len(session.index.get_all_relations()) == 0

        sync_file(
            session.workspace.relations_file,
            session.index,
            session.provider,
            session.config.chunking,
        )
        assert len(session.index.get_all_relations()) == 1
        assert session.index.get_all_relations()[0]["subject"] == "Alice"

    def test_sync_workspace_syncs_relations(self, session):
        """session.sync() on a workspace containing RELATIONS.md syncs relations."""
        _write_relations(session, "- [Bob] --knows--> [Carol]\n")
        assert len(session.index.get_all_relations()) == 0

        session.sync()
        assert len(session.index.get_all_relations()) == 1
        assert session.index.get_all_relations()[0]["subject"] == "Bob"

    def test_sync_file_removes_deleted_relation(self, session):
        """After removing a line from RELATIONS.md, sync_file removes the SQLite row."""
        from groundmemory.core.sync import sync_file

        # First sync: add two relations
        _write_relations(session, "- [A] --b--> [C]\n- [D] --e--> [F]\n")
        sync_file(
            session.workspace.relations_file,
            session.index,
            session.provider,
            session.config.chunking,
        )
        assert len(session.index.get_all_relations()) == 2

        # Remove one, sync again
        _write_relations(session, "- [A] --b--> [C]\n")
        sync_file(
            session.workspace.relations_file,
            session.index,
            session.provider,
            session.config.chunking,
        )
        assert len(session.index.get_all_relations()) == 1

    def test_sync_non_relations_file_does_not_touch_relations_table(self, session):
        """sync_file on USER.md must not alter the relations table."""
        from groundmemory.core.sync import sync_file

        session.workspace.user_file.write_text("Some user info.\n", encoding="utf-8")
        _write_relations(session, "- [Alice] --leads--> [Team]\n")
        sync_relations_from_file(session.workspace.relations_file, session.index)
        assert len(session.index.get_all_relations()) == 1

        sync_file(
            session.workspace.user_file,
            session.index,
            session.provider,
            session.config.chunking,
        )
        # Relations table must be unchanged
        assert len(session.index.get_all_relations()) == 1


# ===========================================================================
# 10. BootstrapConfig.sync_relations_on_bootstrap - config loading
# ===========================================================================


class TestBootstrapConfigDefault:
    def test_default_is_false(self):
        cfg = BootstrapConfig()
        assert cfg.sync_relations_on_bootstrap is False

    def test_explicit_true(self):
        cfg = BootstrapConfig(sync_relations_on_bootstrap=True)
        assert cfg.sync_relations_on_bootstrap is True

    def test_explicit_false(self):
        cfg = BootstrapConfig(sync_relations_on_bootstrap=False)
        assert cfg.sync_relations_on_bootstrap is False

    def test_env_var_sets_true(self, monkeypatch):
        """groundmemory_BOOTSTRAP__SYNC_RELATIONS_ON_BOOTSTRAP=true must be honoured."""
        monkeypatch.setenv("groundmemory_BOOTSTRAP__SYNC_RELATIONS_ON_BOOTSTRAP", "true")
        import groundmemory.config as _cfg_mod

        monkeypatch.setattr(_cfg_mod, "_load_yaml_config", lambda filename="groundmemory.yaml": {})
        cfg = groundmemoryConfig.auto()
        assert cfg.bootstrap.sync_relations_on_bootstrap is True

    def test_env_var_sets_false(self, monkeypatch):
        monkeypatch.setenv("groundmemory_BOOTSTRAP__SYNC_RELATIONS_ON_BOOTSTRAP", "false")
        import groundmemory.config as _cfg_mod

        monkeypatch.setattr(_cfg_mod, "_load_yaml_config", lambda filename="groundmemory.yaml": {})
        cfg = groundmemoryConfig.auto()
        assert cfg.bootstrap.sync_relations_on_bootstrap is False
