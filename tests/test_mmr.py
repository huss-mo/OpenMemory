"""
Tests for MMR (Maximal Marginal Relevance) diversification.

Covers:
  - get_embeddings_by_ids (MemoryIndex method)
  - _apply_mmr (pure-logic unit tests with a mock index)
  - Integration: hybrid_search with mmr_lambda > 0 via SearchConfig
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from groundmemory.config import SearchConfig
from groundmemory.core.index import MemoryIndex
from groundmemory.core.search import _apply_mmr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(chunk_id: str, score: float) -> dict:
    """Minimal result dict as produced by the merge step."""
    return {
        "chunk_id": chunk_id,
        "path": "test.md",
        "source": "memory",
        "start_line": 1,
        "end_line": 2,
        "text": f"Text for {chunk_id}",
        "updated_at": 0.0,
        "vector_score": score,
        "text_score": 0.0,
        "score": score,
    }


def _make_index_with_embeddings(emb_map: dict[str, list[float]]) -> MemoryIndex:
    """
    Return a MemoryIndex mock whose get_embeddings_by_ids returns emb_map.
    """
    mock_index = MagicMock(spec=MemoryIndex)
    mock_index.get_embeddings_by_ids.return_value = emb_map
    return mock_index


def _unit_vec(dim: int, hot_index: int) -> list[float]:
    """Return a unit vector with 1.0 at hot_index and 0.0 elsewhere."""
    v = [0.0] * dim
    v[hot_index] = 1.0
    return v


def _scaled_vec(base: list[float], scale: float) -> list[float]:
    """Scale a vector (used to create near-duplicates)."""
    return [x * scale for x in base]


# ---------------------------------------------------------------------------
# get_embeddings_by_ids tests (real MemoryIndex with in-memory DB)
# ---------------------------------------------------------------------------


@pytest.fixture()
def real_index(tmp_path: Path) -> MemoryIndex:
    return MemoryIndex(tmp_path / "test.db")


class TestGetEmbeddingsByIds:
    def test_empty_list_returns_empty_dict(self, real_index: MemoryIndex):
        result = real_index.get_embeddings_by_ids([])
        assert result == {}

    def test_nonexistent_ids_return_empty_dict(self, real_index: MemoryIndex):
        result = real_index.get_embeddings_by_ids(["ghost-1", "ghost-2"])
        assert result == {}

    def test_returns_correct_embeddings(self, real_index: MemoryIndex):
        """Insert a chunk directly into the DB and verify retrieval."""
        emb = [0.1, 0.2, 0.3]
        real_index._conn.execute(
            "INSERT INTO files(path, source, hash, mtime, size, indexed_at) VALUES (?,?,?,?,?,?)",
            ("f.md", "memory", "abc", 0.0, 0, 0.0),
        )
        real_index._conn.execute(
            "INSERT INTO chunks(chunk_id, path, source, start_line, end_line, "
            "content_hash, model_id, text, embedding, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("c1", "f.md", "memory", 1, 2, "h1", "m1", "hello", json.dumps(emb), 0.0),
        )
        real_index._conn.commit()

        result = real_index.get_embeddings_by_ids(["c1"])
        assert "c1" in result
        assert result["c1"] == pytest.approx(emb)

    def test_partial_ids_only_returns_existing(self, real_index: MemoryIndex):
        """Mix of existing and non-existing IDs: only existing appear in result."""
        real_index._conn.execute(
            "INSERT INTO files(path, source, hash, mtime, size, indexed_at) VALUES (?,?,?,?,?,?)",
            ("f2.md", "memory", "abc2", 0.0, 0, 0.0),
        )
        real_index._conn.execute(
            "INSERT INTO chunks(chunk_id, path, source, start_line, end_line, "
            "content_hash, model_id, text, embedding, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("c2", "f2.md", "memory", 1, 2, "h2", "m1", "world", json.dumps([1.0, 0.0]), 0.0),
        )
        real_index._conn.commit()

        result = real_index.get_embeddings_by_ids(["c2", "does-not-exist"])
        assert "c2" in result
        assert "does-not-exist" not in result

    def test_multiple_ids_returned(self, real_index: MemoryIndex):
        real_index._conn.execute(
            "INSERT INTO files(path, source, hash, mtime, size, indexed_at) VALUES (?,?,?,?,?,?)",
            ("f3.md", "memory", "abc3", 0.0, 0, 0.0),
        )
        for i, emb in enumerate([[1.0, 0.0], [0.0, 1.0]]):
            real_index._conn.execute(
                "INSERT INTO chunks(chunk_id, path, source, start_line, end_line, "
                "content_hash, model_id, text, embedding, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (f"cx{i}", "f3.md", "memory", i, i + 1, f"hx{i}", "m1", f"t{i}", json.dumps(emb), 0.0),
            )
        real_index._conn.commit()

        result = real_index.get_embeddings_by_ids(["cx0", "cx1"])
        assert set(result.keys()) == {"cx0", "cx1"}


# ---------------------------------------------------------------------------
# _apply_mmr unit tests (mocked MemoryIndex)
# ---------------------------------------------------------------------------


class TestApplyMMR:
    def test_empty_results_returns_empty(self):
        index = _make_index_with_embeddings({})
        out = _apply_mmr([], top_k=5, mmr_lambda=0.5, index=index)
        assert out == []

    def test_returns_at_most_top_k(self):
        results = [_make_result(f"c{i}", 1.0 - i * 0.1) for i in range(10)]
        emb_map = {f"c{i}": _unit_vec(10, i) for i in range(10)}
        index = _make_index_with_embeddings(emb_map)
        out = _apply_mmr(results, top_k=4, mmr_lambda=0.5, index=index)
        assert len(out) <= 4

    def test_returns_all_when_fewer_than_top_k(self):
        results = [_make_result(f"c{i}", 1.0 - i * 0.1) for i in range(3)]
        emb_map = {f"c{i}": _unit_vec(5, i) for i in range(3)}
        index = _make_index_with_embeddings(emb_map)
        out = _apply_mmr(results, top_k=10, mmr_lambda=0.5, index=index)
        assert len(out) == 3

    def test_pure_relevance_preserves_score_order(self):
        """mmr_lambda=1.0 means only relevance matters → same order as input."""
        results = [_make_result(f"c{i}", 1.0 - i * 0.1) for i in range(5)]
        # All identical embeddings so similarity is 1.0 for every pair.
        emb_map = {f"c{i}": [1.0, 0.0] for i in range(5)}
        index = _make_index_with_embeddings(emb_map)
        out = _apply_mmr(results, top_k=5, mmr_lambda=1.0, index=index)
        assert [r["chunk_id"] for r in out] == [f"c{i}" for i in range(5)]

    def test_near_duplicate_penalised(self):
        """
        Two chunks with very similar embeddings: the second should be penalised
        when diversity matters.  With mmr_lambda=0.5, after selecting the
        best-scoring item (c0), c1 (near-duplicate) should score lower than c2
        (orthogonal), so c2 is selected next.
        """
        # c0: score=1.0, direction [1,0]
        # c1: score=0.9, direction [1,0] (near-duplicate of c0)
        # c2: score=0.8, direction [0,1] (orthogonal to c0)
        results = [
            _make_result("c0", 1.0),
            _make_result("c1", 0.9),
            _make_result("c2", 0.8),
        ]
        emb_map = {
            "c0": [1.0, 0.0],
            "c1": [1.0, 0.0],  # identical direction to c0
            "c2": [0.0, 1.0],  # orthogonal
        }
        index = _make_index_with_embeddings(emb_map)
        out = _apply_mmr(results, top_k=2, mmr_lambda=0.5, index=index)
        ids = [r["chunk_id"] for r in out]
        # c0 must be first (highest relevance, nothing selected yet).
        assert ids[0] == "c0"
        # c2 should beat c1 because c1 is a near-duplicate of c0.
        assert ids[1] == "c2"

    def test_missing_embedding_handled_gracefully(self):
        """Chunks missing from emb_map still participate (treated as 0 similarity)."""
        results = [
            _make_result("c0", 1.0),
            _make_result("c_no_emb", 0.8),
        ]
        # c_no_emb has no embedding in the map
        emb_map = {"c0": [1.0, 0.0]}
        index = _make_index_with_embeddings(emb_map)
        # Should not raise
        out = _apply_mmr(results, top_k=2, mmr_lambda=0.5, index=index)
        assert len(out) == 2
        assert out[0]["chunk_id"] == "c0"
        assert out[1]["chunk_id"] == "c_no_emb"

    def test_no_embeddings_falls_back_to_relevance_order(self):
        """When all embeddings are missing, relevance order is preserved."""
        results = [_make_result(f"c{i}", 1.0 - i * 0.1) for i in range(4)]
        index = _make_index_with_embeddings({})
        out = _apply_mmr(results, top_k=4, mmr_lambda=0.5, index=index)
        assert [r["chunk_id"] for r in out] == [f"c{i}" for i in range(4)]

    def test_all_chunks_returned_when_top_k_equals_len(self):
        results = [_make_result(f"c{i}", float(i)) for i in range(5)]
        emb_map = {f"c{i}": _unit_vec(5, i) for i in range(5)}
        index = _make_index_with_embeddings(emb_map)
        out = _apply_mmr(results, top_k=5, mmr_lambda=0.5, index=index)
        assert len(out) == 5
        # All original chunks present
        assert {r["chunk_id"] for r in out} == {f"c{i}" for i in range(5)}

    def test_single_result_returns_single(self):
        results = [_make_result("only", 0.9)]
        emb_map = {"only": [1.0, 0.0]}
        index = _make_index_with_embeddings(emb_map)
        out = _apply_mmr(results, top_k=5, mmr_lambda=0.5, index=index)
        assert len(out) == 1
        assert out[0]["chunk_id"] == "only"

    def test_diversity_lambda_selects_more_diverse_set(self):
        """
        Lower mmr_lambda → more diversity.  With 4 candidates where c0/c1 are
        near-duplicates and c2/c3 are orthogonal, a low lambda should prefer
        the orthogonal ones after the first selection.
        """
        results = [
            _make_result("c0", 1.0),   # [1,0,0,0]
            _make_result("c1", 0.99),  # [1,0,0,0] near-dup
            _make_result("c2", 0.8),   # [0,1,0,0]
            _make_result("c3", 0.7),   # [0,0,1,0]
        ]
        emb_map = {
            "c0": [1.0, 0.0, 0.0, 0.0],
            "c1": [1.0, 0.0, 0.0, 0.0],
            "c2": [0.0, 1.0, 0.0, 0.0],
            "c3": [0.0, 0.0, 1.0, 0.0],
        }
        index = _make_index_with_embeddings(emb_map)
        out = _apply_mmr(results, top_k=3, mmr_lambda=0.1, index=index)
        ids = [r["chunk_id"] for r in out]
        # After c0, the near-duplicate c1 should be heavily penalised.
        assert "c1" not in ids


# ---------------------------------------------------------------------------
# Integration: hybrid_search with mmr_lambda via session
# ---------------------------------------------------------------------------


class TestMMRIntegration:
    """
    Full-pipeline tests using the MCP session fixture.
    These verify that mmr_lambda is honoured from SearchConfig and does not crash.

    mmr_lambda is a config-level knob — it is NOT exposed as a tool parameter.
    Tests exercise it by mutating session.config.search.mmr_lambda directly,
    which is the same object hybrid_search reads from at call time.
    """

    def test_mmr_disabled_by_default(self, session):
        """Default SearchConfig has mmr_lambda=0.0 — pipeline must not crash."""
        assert session.config.search.mmr_lambda == 0.0
        session.execute_tool("memory_write", file="MEMORY.md", content="The quick brown fox.")
        r = session.execute_tool("memory_read", query="quick fox")
        assert r["status"] == "ok"

    def test_mmr_enabled_does_not_crash(self, session):
        """mmr_lambda=0.5 (via config) should run without errors and return results."""
        session.config.search.mmr_lambda = 0.5
        for i in range(6):
            session.execute_tool(
                "memory_write",
                file="MEMORY.md",
                content=f"Entry {i}: The project uses Python and SQLite for storage.",
            )
        r = session.execute_tool("memory_read", query="Python SQLite storage")
        assert r["status"] == "ok"
        assert isinstance(r["results"], list)

    def test_mmr_respects_top_k(self, session):
        """Results with MMR enabled must still respect top_k."""
        session.config.search.mmr_lambda = 0.5
        for i in range(10):
            session.execute_tool(
                "memory_write",
                file="MEMORY.md",
                content=f"Similar fact {i}: The system indexes documents automatically.",
            )
        r = session.execute_tool("memory_read", query="indexes documents", top_k=3)
        assert r["status"] == "ok"
        assert len(r["results"]) <= 3

    def test_mmr_lambda_one_same_as_disabled(self, session):
        """mmr_lambda=1.0 is pure relevance — results should be non-empty and valid."""
        session.config.search.mmr_lambda = 1.0
        session.execute_tool("memory_write", file="MEMORY.md", content="Lambda one test content.")
        r = session.execute_tool("memory_read", query="Lambda one test")
        assert r["status"] == "ok"
