"""
Comprehensive tests for the memory compaction feature.

Covers:
  - token_counter (approx + tiktoken)
  - backup: create, list, parse_spec, restore
  - memory_compact tool (success, validation errors)
  - injector: compaction notice injection
  - session.bootstrap(): backup trigger, backup_name propagation
  - BootstrapConfig compaction fields
  - groundmemory --restore CLI helpers
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from groundmemory.config import BootstrapConfig, EmbeddingConfig, groundmemoryConfig
from groundmemory.session import MemorySession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(tmp_path: Path, **bootstrap_kwargs) -> MemorySession:
    bootstrap = BootstrapConfig(**bootstrap_kwargs) if bootstrap_kwargs else BootstrapConfig()
    cfg = groundmemoryConfig(
        root_dir=tmp_path,
        workspace="test",
        embedding=EmbeddingConfig(provider="none"),
        bootstrap=bootstrap,
    )
    return MemorySession.create("test", config=cfg)


# ===========================================================================
# 1. Token counter
# ===========================================================================


class TestTokenCounter:
    def test_approx_empty_string(self):
        from groundmemory.bootstrap.token_counter import count_tokens
        assert count_tokens("", method="approx") == 1  # max(1, 0//4)

    def test_approx_short_text(self):
        from groundmemory.bootstrap.token_counter import count_tokens
        text = "a" * 400
        assert count_tokens(text, method="approx") == 100

    def test_approx_longer_text(self):
        from groundmemory.bootstrap.token_counter import count_tokens
        text = "hello world " * 100  # 1200 chars
        result = count_tokens(text, method="approx")
        assert result == 300

    def test_tiktoken_returns_positive_int(self):
        """tiktoken must return a positive integer for non-empty text."""
        from groundmemory.bootstrap.token_counter import count_tokens
        text = "The quick brown fox jumps over the lazy dog."
        result = count_tokens(text, method="tiktoken")
        assert isinstance(result, int)
        assert result > 0

    def test_tiktoken_longer_than_approx_ratio(self):
        """tiktoken result should be in a sane range relative to approx."""
        from groundmemory.bootstrap.token_counter import count_tokens
        text = "Hello, how are you today? " * 50
        approx = count_tokens(text, method="approx")
        tiktoken_result = count_tokens(text, method="tiktoken")
        # tiktoken tends to be slightly more than approx for English text
        # but should be within 2x
        assert 0.5 * approx < tiktoken_result < 2 * approx

    def test_tiktoken_fallback_when_not_installed(self, monkeypatch):
        """If tiktoken import fails, fall back to approx silently."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("tiktoken not available")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from groundmemory.bootstrap import token_counter
        # Reload to clear cached import
        import importlib
        importlib.reload(token_counter)

        text = "a" * 400
        result = token_counter.count_tokens(text, method="tiktoken")
        assert result == 100  # falls back to approx

        importlib.reload(token_counter)  # restore

    def test_default_method_is_approx(self):
        from groundmemory.bootstrap.token_counter import count_tokens
        text = "x" * 800
        assert count_tokens(text) == count_tokens(text, method="approx")


# ===========================================================================
# 2. Backup: create_backup
# ===========================================================================


class TestCreateBackup:
    def test_creates_zip_file(self, tmp_path):
        from groundmemory.core.backup import create_backup
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "MEMORY.md").write_text("# Memory\n", encoding="utf-8")
        (ws / ".index").mkdir()

        archive = create_backup(ws)
        assert archive.exists()
        assert archive.suffix == ".zip"

    def test_archive_in_backups_subdir(self, tmp_path):
        from groundmemory.core.backup import create_backup
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "MEMORY.md").write_text("content", encoding="utf-8")
        (ws / ".index").mkdir()

        archive = create_backup(ws)
        assert archive.parent.name == "backups"

    def test_archive_name_format(self, tmp_path):
        from groundmemory.core.backup import create_backup
        import re
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / ".index").mkdir()

        archive = create_backup(ws)
        assert re.match(r"\d{4}-\d{2}-\d{2}_\d{6}", archive.stem)

    def test_archive_contains_md_files(self, tmp_path):
        from groundmemory.core.backup import create_backup
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "MEMORY.md").write_text("# Memory\nsome content", encoding="utf-8")
        (ws / "USER.md").write_text("# User\n", encoding="utf-8")
        (ws / ".index").mkdir()

        archive = create_backup(ws)
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
        assert "MEMORY.md" in names
        assert "USER.md" in names

    def test_archive_contains_daily_logs(self, tmp_path):
        from groundmemory.core.backup import create_backup
        ws = tmp_path / "ws"
        ws.mkdir()
        daily = ws / "daily"
        daily.mkdir()
        (daily / "2026-04-08.md").write_text("log entry", encoding="utf-8")
        (ws / ".index").mkdir()

        archive = create_backup(ws)
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
        assert any("2026-04-08.md" in n for n in names)

    def test_archive_contains_db(self, tmp_path):
        from groundmemory.core.backup import create_backup
        ws = tmp_path / "ws"
        ws.mkdir()
        idx = ws / ".index"
        idx.mkdir()
        (idx / "memory.db").write_bytes(b"SQLite data")

        archive = create_backup(ws)
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
        assert any("memory.db" in n for n in names)

    def test_backup_excludes_backup_dir(self, tmp_path):
        from groundmemory.core.backup import create_backup
        ws = tmp_path / "ws"
        ws.mkdir()
        backups = ws / "backups"
        backups.mkdir()
        (backups / "old.zip").write_bytes(b"old backup")
        (ws / ".index").mkdir()

        archive = create_backup(ws)
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
        # The backup directory itself should not be inside the archive
        assert not any("backups" in n for n in names)


# ===========================================================================
# 3. Backup: list_backups + parse_spec
# ===========================================================================


class TestBackupListAndParse:
    def _make_backups(self, workspace: Path, stems: list[str]) -> list[Path]:
        from groundmemory.core.backup import backup_dir
        bd = backup_dir(workspace)
        paths = []
        for stem in stems:
            p = bd / f"{stem}.zip"
            p.write_bytes(b"")
            paths.append(p)
        return sorted(paths)

    def test_list_backups_empty(self, tmp_path):
        from groundmemory.core.backup import list_backups
        ws = tmp_path / "ws"
        ws.mkdir()
        assert list_backups(ws) == []

    def test_list_backups_sorted(self, tmp_path):
        from groundmemory.core.backup import list_backups
        ws = tmp_path / "ws"
        ws.mkdir()
        stems = ["2026-04-08_120000", "2026-04-07_080000", "2026-04-08_090000"]
        self._make_backups(ws, stems)
        result = list_backups(ws)
        assert [r.stem for r in result] == sorted(stems)

    def test_parse_spec_minus_one(self, tmp_path):
        from groundmemory.core.backup import list_backups, parse_spec
        ws = tmp_path / "ws"
        ws.mkdir()
        self._make_backups(ws, ["2026-04-07_080000", "2026-04-08_120000"])
        backups = list_backups(ws)
        result = parse_spec("-1", backups)
        assert result is not None
        assert result.stem == "2026-04-08_120000"

    def test_parse_spec_minus_two(self, tmp_path):
        from groundmemory.core.backup import list_backups, parse_spec
        ws = tmp_path / "ws"
        ws.mkdir()
        self._make_backups(ws, ["2026-04-07_080000", "2026-04-08_120000"])
        backups = list_backups(ws)
        result = parse_spec("-2", backups)
        assert result is not None
        assert result.stem == "2026-04-07_080000"

    def test_parse_spec_exact_timestamp(self, tmp_path):
        from groundmemory.core.backup import list_backups, parse_spec
        ws = tmp_path / "ws"
        ws.mkdir()
        self._make_backups(ws, ["2026-04-08_165530", "2026-04-08_180000"])
        backups = list_backups(ws)
        result = parse_spec("2026-04-08_165530", backups)
        assert result is not None
        assert result.stem == "2026-04-08_165530"

    def test_parse_spec_date_single_match(self, tmp_path):
        from groundmemory.core.backup import list_backups, parse_spec
        ws = tmp_path / "ws"
        ws.mkdir()
        self._make_backups(ws, ["2026-04-07_080000", "2026-04-08_165530"])
        backups = list_backups(ws)
        result = parse_spec("2026-04-08", backups)
        assert result is not None
        assert result.stem == "2026-04-08_165530"

    def test_parse_spec_date_ambiguous_returns_none(self, tmp_path):
        from groundmemory.core.backup import list_backups, parse_spec
        ws = tmp_path / "ws"
        ws.mkdir()
        self._make_backups(ws, ["2026-04-08_090000", "2026-04-08_165530"])
        backups = list_backups(ws)
        result = parse_spec("2026-04-08", backups)
        assert result is None

    def test_parse_spec_no_match_returns_none(self, tmp_path):
        from groundmemory.core.backup import list_backups, parse_spec
        ws = tmp_path / "ws"
        ws.mkdir()
        self._make_backups(ws, ["2026-04-07_080000"])
        backups = list_backups(ws)
        assert parse_spec("2026-04-08", backups) is None

    def test_parse_spec_empty_list(self):
        from groundmemory.core.backup import parse_spec
        assert parse_spec("-1", []) is None


# ===========================================================================
# 4. Backup: restore_backup
# ===========================================================================


class TestRestoreBackup:
    def test_restore_overwrites_files(self, tmp_path):
        from groundmemory.core.backup import create_backup, restore_backup

        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "MEMORY.md").write_text("original content", encoding="utf-8")
        (ws / ".index").mkdir()

        archive = create_backup(ws)

        # Modify the file after backup
        (ws / "MEMORY.md").write_text("modified content", encoding="utf-8")

        restore_backup(archive, ws)
        assert (ws / "MEMORY.md").read_text(encoding="utf-8") == "original content"

    def test_restore_creates_index_dir(self, tmp_path):
        from groundmemory.core.backup import create_backup, restore_backup

        ws = tmp_path / "ws"
        ws.mkdir()
        idx = ws / ".index"
        idx.mkdir()
        (idx / "memory.db").write_bytes(b"db data")

        archive = create_backup(ws)

        # Remove index dir
        import shutil
        shutil.rmtree(idx)

        restore_backup(archive, ws)
        assert idx.exists()


# ===========================================================================
# 5. memory_compact tool
# ===========================================================================


class TestMemoryCompactTool:
    def test_compact_memory_md_success(self, tmp_path):
        s = _make_session(tmp_path, compaction_tiers=["MEMORY.md"])
        try:
            s.workspace.first_run_file.write_text("", encoding="utf-8")
            s.execute_tool("memory_write", file="MEMORY.md", content="Old content line 1.\nOld content line 2.\n")
            result = s.execute_tool("memory_compact", tier="MEMORY.md", content="Compacted: Old content.")
            assert result["status"] == "ok"
            assert result["tier"] == "MEMORY.md"
        finally:
            s.close()

    def test_compact_overwrites_file(self, tmp_path):
        s = _make_session(tmp_path, compaction_tiers=["MEMORY.md"])
        try:
            s.workspace.first_run_file.write_text("", encoding="utf-8")
            s.execute_tool("memory_write", file="MEMORY.md", content="Line 1.\nLine 2.\nLine 3.\n")
            s.execute_tool("memory_compact", tier="MEMORY.md", content="Compacted.")
            text = s.workspace.memory_file.read_text(encoding="utf-8")
            assert text.strip() == "Compacted."
            assert "Line 1" not in text
        finally:
            s.close()

    def test_compact_rejects_unknown_tier(self, tmp_path):
        s = _make_session(tmp_path, compaction_tiers=["MEMORY.md"])
        try:
            result = s.execute_tool("memory_compact", tier="daily", content="x")
            assert result["status"] == "error"
            assert "not a compactable tier" in result["message"].lower()
        finally:
            s.close()

    def test_compact_rejects_tier_not_in_config(self, tmp_path):
        s = _make_session(tmp_path, compaction_tiers=["MEMORY.md"])
        try:
            result = s.execute_tool("memory_compact", tier="USER.md", content="x")
            assert result["status"] == "error"
            assert "not in the configured" in result["message"].lower()
        finally:
            s.close()

    def test_compact_rejects_empty_content(self, tmp_path):
        s = _make_session(tmp_path, compaction_tiers=["MEMORY.md"])
        try:
            result = s.execute_tool("memory_compact", tier="MEMORY.md", content="")
            assert result["status"] == "error"
        finally:
            s.close()

    def test_compact_user_md_when_configured(self, tmp_path):
        s = _make_session(tmp_path, compaction_tiers=["MEMORY.md", "USER.md"])
        try:
            result = s.execute_tool("memory_compact", tier="USER.md", content="Compacted user profile.")
            assert result["status"] == "ok"
        finally:
            s.close()

    def test_compact_agents_md_when_configured(self, tmp_path):
        s = _make_session(tmp_path, compaction_tiers=["AGENTS.md"])
        try:
            result = s.execute_tool("memory_compact", tier="AGENTS.md", content="Compacted agents.")
            assert result["status"] == "ok"
        finally:
            s.close()

    def test_compact_chars_written_returned(self, tmp_path):
        s = _make_session(tmp_path, compaction_tiers=["MEMORY.md"])
        try:
            content = "Compacted memory content."
            result = s.execute_tool("memory_compact", tier="MEMORY.md", content=content)
            assert result["status"] == "ok"
            assert result["chars_written"] == len(content)
        finally:
            s.close()


# ===========================================================================
# 6. Bootstrap compaction notice injection
# ===========================================================================


class TestBootstrapCompactionNotice:
    def test_no_notice_when_threshold_zero(self, tmp_path):
        """Compaction is disabled by default (threshold=0)."""
        s = _make_session(tmp_path, compaction_token_threshold=0)
        try:
            s.workspace.first_run_file.write_text("", encoding="utf-8")
            s.execute_tool("memory_write", file="MEMORY.md", content="Some facts. " * 100)
            result = s.bootstrap()
            assert "Compaction Required" not in result
        finally:
            s.close()

    def test_notice_injected_when_above_threshold(self, tmp_path):
        """When bootstrap tokens > threshold a compaction notice must be present."""
        s = _make_session(
            tmp_path,
            compaction_token_threshold=1,  # very low to guarantee trigger
            compaction_tiers=["MEMORY.md"],
        )
        try:
            s.workspace.first_run_file.write_text("", encoding="utf-8")
            s.execute_tool("memory_write", file="MEMORY.md", content="Some content.")
            result = s.bootstrap()
            assert "Compaction Required" in result
        finally:
            s.close()

    def test_notice_names_configured_tiers(self, tmp_path):
        s = _make_session(
            tmp_path,
            compaction_token_threshold=1,
            compaction_tiers=["MEMORY.md", "USER.md"],
        )
        try:
            s.workspace.first_run_file.write_text("", encoding="utf-8")
            s.execute_tool("memory_write", file="MEMORY.md", content="content")
            result = s.bootstrap()
            assert "MEMORY.md" in result
            assert "USER.md" in result
        finally:
            s.close()

    def test_no_notice_when_below_threshold(self, tmp_path):
        s = _make_session(
            tmp_path,
            compaction_token_threshold=100_000,  # very high, never triggered
            compaction_tiers=["MEMORY.md"],
        )
        try:
            s.workspace.first_run_file.write_text("", encoding="utf-8")
            s.execute_tool("memory_write", file="MEMORY.md", content="short")
            result = s.bootstrap()
            assert "Compaction Required" not in result
        finally:
            s.close()


# ===========================================================================
# 7. Session bootstrap: backup is taken when threshold exceeded
# ===========================================================================


class TestSessionBootstrapBackup:
    def test_backup_created_when_above_threshold(self, tmp_path):
        from groundmemory.core.backup import list_backups
        s = _make_session(
            tmp_path,
            compaction_token_threshold=1,
            compaction_tiers=["MEMORY.md"],
        )
        try:
            s.workspace.first_run_file.write_text("", encoding="utf-8")
            s.execute_tool("memory_write", file="MEMORY.md", content="content")
            s.bootstrap()
            backups = list_backups(s.workspace.workspace_path)
            assert len(backups) == 1
        finally:
            s.close()

    def test_no_backup_when_threshold_zero(self, tmp_path):
        from groundmemory.core.backup import list_backups
        s = _make_session(tmp_path, compaction_token_threshold=0)
        try:
            s.workspace.first_run_file.write_text("", encoding="utf-8")
            s.execute_tool("memory_write", file="MEMORY.md", content="content " * 500)
            s.bootstrap()
            backups = list_backups(s.workspace.workspace_path)
            assert len(backups) == 0
        finally:
            s.close()

    def test_backup_taken_on_repeated_bootstrap(self, tmp_path):
        """Multiple bootstrap calls above threshold each attempt a backup.
        Same-second calls may produce the same filename (overwrite); the
        result is at least 1 backup in the backups directory."""
        from groundmemory.core.backup import list_backups
        s = _make_session(
            tmp_path,
            compaction_token_threshold=1,
            compaction_tiers=["MEMORY.md"],
        )
        try:
            s.workspace.first_run_file.write_text("", encoding="utf-8")
            s.execute_tool("memory_write", file="MEMORY.md", content="content")
            s.bootstrap()
            s.bootstrap()
            backups = list_backups(s.workspace.workspace_path)
            assert len(backups) >= 1
        finally:
            s.close()

    def test_backup_created_and_notice_injected(self, tmp_path):
        """When threshold is exceeded: backup exists on disk and notice is in the prompt."""
        from groundmemory.core.backup import list_backups
        s = _make_session(
            tmp_path,
            compaction_token_threshold=1,
            compaction_tiers=["MEMORY.md"],
        )
        try:
            s.workspace.first_run_file.write_text("", encoding="utf-8")
            s.execute_tool("memory_write", file="MEMORY.md", content="content")
            result = s.bootstrap()
            # Backup file exists
            assert len(list_backups(s.workspace.workspace_path)) == 1
            # Compaction notice is in the prompt
            assert "Compaction Required" in result
            # Backup name is NOT in the prompt (it's only logged to server output)
            import re
            assert not re.search(r"\d{4}-\d{2}-\d{2}_\d{6}", result)
        finally:
            s.close()


# ===========================================================================
# 8. BootstrapConfig compaction field defaults
# ===========================================================================


class TestBootstrapConfigCompactionDefaults:
    def test_threshold_default_is_zero(self):
        cfg = BootstrapConfig()
        assert cfg.compaction_token_threshold == 0

    def test_counter_default_is_approx(self):
        cfg = BootstrapConfig()
        assert cfg.compaction_token_counter == "approx"

    def test_tiers_default_is_memory_md(self):
        cfg = BootstrapConfig()
        assert cfg.compaction_tiers == ["MEMORY.md"]

    def test_threshold_can_be_set(self):
        cfg = BootstrapConfig(compaction_token_threshold=6000)
        assert cfg.compaction_token_threshold == 6000

    def test_counter_can_be_tiktoken(self):
        cfg = BootstrapConfig(compaction_token_counter="tiktoken")
        assert cfg.compaction_token_counter == "tiktoken"

    def test_tiers_can_be_extended(self):
        cfg = BootstrapConfig(compaction_tiers=["MEMORY.md", "USER.md"])
        assert "USER.md" in cfg.compaction_tiers

    def test_env_var_sets_threshold(self, monkeypatch):
        monkeypatch.setenv("GROUNDMEMORY_BOOTSTRAP__COMPACTION_TOKEN_THRESHOLD", "5000")
        cfg = BootstrapConfig()
        assert cfg.compaction_token_threshold == 5000

    def test_env_var_sets_counter(self, monkeypatch):
        monkeypatch.setenv("GROUNDMEMORY_BOOTSTRAP__COMPACTION_TOKEN_COUNTER", "tiktoken")
        cfg = BootstrapConfig()
        assert cfg.compaction_token_counter == "tiktoken"


# ===========================================================================
# 9. memory_compact registered in tool registry
# ===========================================================================


class TestMemoryCompactRegistered:
    def test_memory_compact_in_tool_runners(self, tmp_path):
        s = _make_session(tmp_path)
        try:
            assert "memory_compact" in s._tool_runners
        finally:
            s.close()

    def test_memory_compact_schema_has_required_fields(self, tmp_path):
        from groundmemory.tools.memory_compact import SCHEMA
        assert SCHEMA["name"] == "memory_compact"
        props = SCHEMA["parameters"]["properties"]
        assert "tier" in props
        assert "content" in props
        assert SCHEMA["parameters"]["required"] == ["tier", "content"]
