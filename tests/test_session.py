"""Tests for MemorySession lifecycle, config loading, bootstrap, and compaction."""
from __future__ import annotations

import pytest

from groundmemory.config import groundmemoryConfig, EmbeddingConfig, MCPConfig, SearchConfig
from groundmemory.session import MemorySession


class TestSessionCreation:
    def test_create_default_session(self, tmp_path):
        cfg = groundmemoryConfig(
            root_dir=tmp_path,
            embedding=EmbeddingConfig(provider="none"),
            mcp=MCPConfig(expose_memory_list=True),
        )
        s = MemorySession.create("default", config=cfg)
        assert s is not None
        s.close()

    def test_create_named_session(self, tmp_path):
        cfg = groundmemoryConfig(
            root_dir=tmp_path,
            embedding=EmbeddingConfig(provider="none"),
            mcp=MCPConfig(expose_memory_list=True),
        )
        s = MemorySession.create("my_project", config=cfg)
        assert "my_project" in str(s.workspace.workspace_path)
        s.close()

    def test_create_uses_auto_config_when_none_given(self, tmp_path, monkeypatch):
        """MemorySession.create() without a config must not raise."""
        monkeypatch.setenv("GROUNDMEMORY_ROOT_DIR", str(tmp_path))
        monkeypatch.setenv("GROUNDMEMORY_EMBEDDING__PROVIDER", "none")
        s = MemorySession.create("auto_test")
        assert s is not None
        s.close()

    def test_context_manager_closes_session(self, tmp_path):
        cfg = groundmemoryConfig(
            root_dir=tmp_path,
            embedding=EmbeddingConfig(provider="none"),
            mcp=MCPConfig(expose_memory_list=True),
        )
        with MemorySession.create("ctx_test", config=cfg) as s:
            r = s.execute_tool("memory_write", file="MEMORY.md", content="CM test.")
            assert r["status"] == "ok"

    def test_two_sessions_are_independent(self, tmp_path):
        cfg = groundmemoryConfig(
            root_dir=tmp_path,
            embedding=EmbeddingConfig(provider="none"),
            mcp=MCPConfig(expose_memory_list=True),
        )
        s1 = MemorySession.create("ws_a", config=cfg)
        s2 = MemorySession.create("ws_b", config=cfg)

        s1.execute_tool("memory_write", file="MEMORY.md", content="Session A content.")
        r = s2.execute_tool("memory_read", query="Session A content")
        # s2 should NOT find s1's content
        assert r["count"] == 0

        s1.close()
        s2.close()

    def test_reopen_same_session_preserves_data(self, tmp_path):
        cfg = groundmemoryConfig(
            root_dir=tmp_path,
            workspace="test",
            embedding=EmbeddingConfig(provider="none"),
            mcp=MCPConfig(expose_memory_list=True),
        )
        s1 = MemorySession.create("persistent", config=cfg)
        s1.execute_tool("memory_write", file="MEMORY.md", content="Persistent fact.")
        s1.close()

        # Reopen the same workspace
        s2 = MemorySession.create("persistent", config=cfg)
        r = s2.execute_tool("memory_read", file="MEMORY.md")
        assert "Persistent fact." in r["content"]
        s2.close()


class TestSessionToolRegistry:
    def test_all_four_core_tools_registered(self, session):
        from groundmemory.tools import TOOL_RUNNERS
        expected = {
            "memory_bootstrap", "memory_read", "memory_write",
            "memory_relate",
        }
        # The 4 core tools must all be registered; memory_list is optional/gated
        assert expected.issubset(set(TOOL_RUNNERS.keys()))

    def test_unknown_tool_returns_error(self, session):
        r = session.execute_tool("nonexistent_tool")
        assert r["status"] == "error"
        assert "unknown" in r["message"].lower() or "nonexistent_tool" in r["message"].lower()


class TestSessionSync:
    def test_sync_returns_dict(self, session):
        result = session.sync()
        assert isinstance(result, dict)

    def test_sync_after_write_indexes_content(self, session):
        session.execute_tool("memory_write", file="MEMORY.md", content="Sync test content.")
        result = session.sync()
        assert isinstance(result, dict)
        # Should have processed files
        assert "added" in result or "updated" in result or "skipped" in result


class TestSessionBootstrap:
    def test_bootstrap_returns_string(self, session):
        result = session.bootstrap()
        assert isinstance(result, str)

    def test_bootstrap_empty_workspace_returns_empty_or_minimal(self, session):
        result = session.bootstrap()
        assert isinstance(result, str)
        # An empty workspace should produce minimal/empty bootstrap

    def test_bootstrap_includes_long_term_memory(self, session):
        # Simulate post-onboarding: FIRST_RUN.md must be empty so MEMORY.md is injected
        session.workspace.first_run_file.write_text("", encoding="utf-8")
        content = "Bootstrap long term fact."
        session.execute_tool("memory_write", file="MEMORY.md", content=content)
        result = session.bootstrap()
        assert content in result

    def test_bootstrap_includes_daily_log(self, session):
        content = "Bootstrap daily note."
        session.execute_tool("memory_write", file="daily", content=content)
        result = session.bootstrap()
        assert content in result

    def test_bootstrap_includes_relations(self, session):
        session.execute_tool(
            "memory_relate", subject="Bootstrap", predicate="tests", object="Relations"
        )
        result = session.bootstrap()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_daily_log_days_default_is_1(self):
        from groundmemory.config import BootstrapConfig
        assert BootstrapConfig().daily_log_days == 1

    def test_daily_log_days_zero_excludes_daily(self, tmp_path):
        """daily_log_days=0 must not inject any daily log even if inject_daily_logs=True."""
        import datetime
        from groundmemory.config import BootstrapConfig, EmbeddingConfig, groundmemoryConfig
        cfg = groundmemoryConfig(
            root_dir=tmp_path,
            embedding=EmbeddingConfig(provider="none"),
            bootstrap=BootstrapConfig(daily_log_days=0),
        )
        s = MemorySession.create("dl_zero", config=cfg)
        try:
            s.execute_tool("memory_write", file="daily", content="Should not appear.")
            result = s.bootstrap()
            assert "Should not appear." not in result
        finally:
            s.close()

    def test_daily_log_days_1_injects_only_today(self, tmp_path):
        """daily_log_days=1 injects today's log but not yesterday's."""
        import datetime
        from groundmemory.config import BootstrapConfig, EmbeddingConfig, groundmemoryConfig
        cfg = groundmemoryConfig(
            root_dir=tmp_path,
            embedding=EmbeddingConfig(provider="none"),
            bootstrap=BootstrapConfig(daily_log_days=1),
        )
        s = MemorySession.create("dl_one", config=cfg)
        try:
            # Write a fake "yesterday" log directly
            yesterday = datetime.date.today() - datetime.timedelta(days=1)
            s.workspace.daily_file(yesterday).write_text("Yesterday entry.", encoding="utf-8")
            s.execute_tool("memory_write", file="daily", content="Today entry.")
            result = s.bootstrap()
            assert "Today entry." in result
            assert "Yesterday entry." not in result
        finally:
            s.close()

    def test_daily_log_days_2_injects_today_and_yesterday(self, tmp_path):
        """daily_log_days=2 injects both today's and yesterday's logs."""
        import datetime
        from groundmemory.config import BootstrapConfig, EmbeddingConfig, groundmemoryConfig
        cfg = groundmemoryConfig(
            root_dir=tmp_path,
            embedding=EmbeddingConfig(provider="none"),
            bootstrap=BootstrapConfig(daily_log_days=2),
        )
        s = MemorySession.create("dl_two", config=cfg)
        try:
            yesterday = datetime.date.today() - datetime.timedelta(days=1)
            s.workspace.daily_file(yesterday).write_text("Yesterday entry.", encoding="utf-8")
            s.execute_tool("memory_write", file="daily", content="Today entry.")
            result = s.bootstrap()
            assert "Today entry." in result
            assert "Yesterday entry." in result
        finally:
            s.close()


class TestConfigFromYaml:
    def test_from_yaml_loads_embedding_provider(self, tmp_path):
        import yaml
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump({
            "embedding": {"provider": "none"},
            "search": {"top_k": 12},
        }))
        cfg = groundmemoryConfig.from_yaml(cfg_file)
        assert cfg.embedding.provider == "none"
        assert cfg.search.top_k == 12

    def test_auto_falls_back_to_defaults_without_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GROUNDMEMORY_EMBEDDING__BASE_URL", raising=False)
        monkeypatch.setenv("GROUNDMEMORY_EMBEDDING__PROVIDER", "none")
        # Patch _load_yaml_config so the project-root groundmemory.yaml is not picked up
        import groundmemory.config as _cfg_module
        monkeypatch.setattr(_cfg_module, "_load_yaml_config", lambda filename="groundmemory.yaml": {})
        cfg = groundmemoryConfig.auto()
        assert cfg.embedding.provider == "none"

    def test_env_overrides_yaml(self, tmp_path, monkeypatch):
        import yaml
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump({
            "embedding": {"provider": "local", "local_model": "all-MiniLM-L6-v2"},
        }))
        monkeypatch.setenv("GROUNDMEMORY_EMBEDDING__PROVIDER", "none")
        monkeypatch.delenv("GROUNDMEMORY_EMBEDDING__BASE_URL", raising=False)
        cfg = groundmemoryConfig.from_yaml(cfg_file)
        # env var should win over YAML
        assert cfg.embedding.provider == "none"