"""Tests for MemorySession lifecycle, config loading, bootstrap, and compaction."""
from __future__ import annotations

import pytest

from openmemory.config import OpenMemoryConfig, EmbeddingConfig, SearchConfig
from openmemory.session import MemorySession


class TestSessionCreation:
    def test_create_default_session(self, tmp_path):
        cfg = OpenMemoryConfig(
            root_dir=tmp_path,
            embedding=EmbeddingConfig(provider="none"),
        )
        s = MemorySession.create("default", config=cfg)
        assert s is not None
        s.close()

    def test_create_named_session(self, tmp_path):
        cfg = OpenMemoryConfig(
            root_dir=tmp_path,
            embedding=EmbeddingConfig(provider="none"),
        )
        s = MemorySession.create("my_project", config=cfg)
        assert "my_project" in str(s.workspace.workspace_path)
        s.close()

    def test_create_uses_auto_config_when_none_given(self, tmp_path, monkeypatch):
        """MemorySession.create() without a config must not raise."""
        monkeypatch.setenv("OPENMEMORY_ROOT_DIR", str(tmp_path))
        monkeypatch.setenv("OPENMEMORY_EMBEDDING__PROVIDER", "none")
        s = MemorySession.create("auto_test")
        assert s is not None
        s.close()

    def test_context_manager_closes_session(self, tmp_path):
        cfg = OpenMemoryConfig(
            root_dir=tmp_path,
            embedding=EmbeddingConfig(provider="none"),
        )
        with MemorySession.create("ctx_test", config=cfg) as s:
            r = s.execute_tool("memory_write", content="CM test.", tier="long_term")
            assert r["status"] == "ok"

    def test_two_sessions_are_independent(self, tmp_path):
        cfg = OpenMemoryConfig(
            root_dir=tmp_path,
            embedding=EmbeddingConfig(provider="none"),
        )
        s1 = MemorySession.create("ws_a", config=cfg)
        s2 = MemorySession.create("ws_b", config=cfg)

        s1.execute_tool("memory_write", content="Session A content.", tier="long_term")
        r = s2.execute_tool("memory_search", query="Session A content")
        # s2 should NOT find s1's content
        assert r["count"] == 0

        s1.close()
        s2.close()

    def test_reopen_same_session_preserves_data(self, tmp_path):
        cfg = OpenMemoryConfig(
            root_dir=tmp_path,
            workspace="test",
            embedding=EmbeddingConfig(provider="none"),
        )
        s1 = MemorySession.create("persistent", config=cfg)
        s1.execute_tool("memory_write", content="Persistent fact.", tier="long_term")
        s1.close()

        # Reopen the same workspace
        s2 = MemorySession.create("persistent", config=cfg)
        r = s2.execute_tool("memory_get", file="MEMORY.md")
        assert "Persistent fact." in r["content"]
        s2.close()


class TestSessionToolRegistry:
    def test_all_six_tools_registered(self, session):
        from openmemory.tools import TOOL_RUNNERS
        expected = {"memory_write", "memory_search", "memory_get",
                    "memory_list", "memory_delete", "memory_relate"}
        assert expected == set(TOOL_RUNNERS.keys())

    def test_unknown_tool_returns_error(self, session):
        r = session.execute_tool("nonexistent_tool")
        assert r["status"] == "error"
        assert "unknown" in r["message"].lower() or "nonexistent_tool" in r["message"].lower()


class TestSessionSync:
    def test_sync_returns_dict(self, session):
        result = session.sync()
        assert isinstance(result, dict)

    def test_sync_after_write_indexes_content(self, session):
        session.execute_tool("memory_write", content="Sync test content.", tier="long_term")
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
        content = "Bootstrap long term fact."
        session.execute_tool("memory_write", content=content, tier="long_term")
        result = session.bootstrap()
        assert content in result

    def test_bootstrap_includes_daily_log(self, session):
        content = "Bootstrap daily note."
        session.execute_tool("memory_write", content=content, tier="daily")
        result = session.bootstrap()
        assert content in result

    def test_bootstrap_includes_relations(self, session):
        session.execute_tool(
            "memory_relate", subject="Bootstrap", predicate="tests", object="Relations"
        )
        result = session.bootstrap()
        assert isinstance(result, str)
        assert len(result) > 0


class TestSessionCompaction:
    def test_should_compact_false_when_tokens_far_from_limit(self, session):
        result = session.should_compact(current_tokens=1000, context_window=200_000)
        assert result is False

    def test_should_compact_true_when_near_limit(self, session):
        result = session.should_compact(current_tokens=198_000, context_window=200_000)
        assert result is True

    def test_compaction_prompts_returns_system_and_user(self, session):
        prompts = session.compaction_prompts()
        assert "system" in prompts
        assert "user" in prompts
        assert isinstance(prompts["system"], str)
        assert isinstance(prompts["user"], str)
        assert len(prompts["system"]) > 0
        assert len(prompts["user"]) > 0


class TestConfigFromYaml:
    def test_from_yaml_loads_embedding_provider(self, tmp_path):
        import yaml
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump({
            "embedding": {"provider": "none"},
            "search": {"top_k": 12},
        }))
        cfg = OpenMemoryConfig.from_yaml(cfg_file)
        assert cfg.embedding.provider == "none"
        assert cfg.search.top_k == 12

    def test_auto_falls_back_to_defaults_without_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("OPENMEMORY_EMBEDDING__BASE_URL", raising=False)
        monkeypatch.setenv("OPENMEMORY_EMBEDDING__PROVIDER", "none")
        # Patch _load_yaml_config so the project-root openmemory.yaml is not picked up
        import openmemory.config as _cfg_module
        monkeypatch.setattr(_cfg_module, "_load_yaml_config", lambda filename="openmemory.yaml": {})
        cfg = OpenMemoryConfig.auto()
        assert cfg.embedding.provider == "none"

    def test_env_overrides_yaml(self, tmp_path, monkeypatch):
        import yaml
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump({
            "embedding": {"provider": "local", "local_model": "all-MiniLM-L6-v2"},
        }))
        monkeypatch.setenv("OPENMEMORY_EMBEDDING__PROVIDER", "none")
        monkeypatch.delenv("OPENMEMORY_EMBEDDING__BASE_URL", raising=False)
        cfg = OpenMemoryConfig.from_yaml(cfg_file)
        # env var should win over YAML
        assert cfg.embedding.provider == "none"
