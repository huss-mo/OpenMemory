"""
Shared pytest fixtures for groundmemory tests.

All tests use provider="none" (BM25-only, no network calls) by default.
Tests that need real embeddings are marked with @pytest.mark.embeddings
and require the LiteLLM proxy to be reachable.
"""
from __future__ import annotations

import uuid
import pytest

from groundmemory.config import (
    groundmemoryConfig,
    EmbeddingConfig,
    SearchConfig,
)
from groundmemory.session import MemorySession


def _make_session(tmp_path, provider="none", **search_kwargs) -> MemorySession:
    """Create an isolated MemorySession backed by a temp directory."""
    cfg = groundmemoryConfig(
        root_dir=tmp_path,
        workspace="test",
        embedding=EmbeddingConfig(provider=provider),
        search=SearchConfig(**search_kwargs),
        expose_memory_list=True,
    )
    name = uuid.uuid4().hex[:8]
    return MemorySession.create(name, config=cfg)


@pytest.fixture()
def session(tmp_path):
    """Fresh BM25-only session, cleaned up after each test."""
    s = _make_session(tmp_path)
    yield s
    s.close()


@pytest.fixture()
def session_factory(tmp_path):
    """Factory fixture - call it to get additional isolated sessions."""
    sessions = []

    def _factory(provider="none", **search_kwargs):
        s = _make_session(tmp_path, provider=provider, **search_kwargs)
        sessions.append(s)
        return s

    yield _factory

    for s in sessions:
        try:
            s.close()
        except Exception:
            pass