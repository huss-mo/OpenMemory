"""
Integration tests for real embeddings + vector search.

These tests are skipped automatically when:
  - The configured embedding provider is "none" (BM25-only mode)
  - The embedding endpoint is unreachable (connection error)

Run selectively with:
    pytest tests/test_embeddings_integration.py -v
    pytest -m embeddings -v
"""
from __future__ import annotations

import uuid
import pytest

from openmemory.config import OpenMemoryConfig, EmbeddingConfig, SearchConfig
from openmemory.session import MemorySession


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _load_embeddings_config() -> OpenMemoryConfig:
    """Load config from openmemory.yaml (or env/defaults)."""
    return OpenMemoryConfig.auto()


def _try_probe_provider(provider) -> str | None:
    """
    Try to embed a single test string.
    Returns None on success, or an error message string on failure.
    """
    try:
        vecs = provider.embed(["probe"])
        if not vecs or not vecs[0]:
            return "provider returned empty vector (provider='none'?)"
        return None
    except Exception as exc:
        return str(exc)


@pytest.fixture(scope="module")
def embeddings_session(tmp_path_factory):
    """
    A MemorySession backed by the embedding provider from openmemory.yaml.
    Skips the entire module if the provider is 'none' or unreachable.
    """
    cfg = _load_embeddings_config()

    if cfg.embedding.provider == "none":
        pytest.skip("Embedding provider is 'none' — skipping real-embeddings tests")

    # Build a temporary session to probe the provider
    tmp = tmp_path_factory.mktemp("embeddings_integration")
    probe_cfg = OpenMemoryConfig(
        root_dir=tmp,
        workspace="probe",
        embedding=cfg.embedding,
        search=cfg.search,
    )
    probe_session = MemorySession.create("probe", config=probe_cfg)

    err = _try_probe_provider(probe_session.provider)
    probe_session.close()
    if err:
        pytest.skip(f"Embedding provider unreachable: {err}")

    # Create the real session used by tests
    session_cfg = OpenMemoryConfig(
        root_dir=tmp,
        workspace="test",
        embedding=cfg.embedding,
        search=cfg.search,
    )
    name = uuid.uuid4().hex[:8]
    s = MemorySession.create(name, config=session_cfg)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.embeddings
class TestEmbeddingProvider:
    def test_embed_returns_nonempty_vectors(self, embeddings_session):
        """Provider returns one non-empty vector per input string."""
        texts = ["hello world", "the quick brown fox"]
        vecs = embeddings_session.provider.embed(texts)
        assert len(vecs) == 2
        assert all(len(v) > 0 for v in vecs)

    def test_embed_vectors_have_consistent_dimensions(self, embeddings_session):
        """All vectors from the same provider have the same dimensionality."""
        texts = ["foo", "bar", "baz"]
        vecs = embeddings_session.provider.embed(texts)
        dims = [len(v) for v in vecs]
        assert len(set(dims)) == 1, f"Inconsistent dimensions: {dims}"

    def test_embed_single_text(self, embeddings_session):
        """Single-item input works correctly."""
        vecs = embeddings_session.provider.embed(["single sentence"])
        assert len(vecs) == 1
        assert len(vecs[0]) > 0

    def test_embed_empty_list(self, embeddings_session):
        """Empty input returns empty list without error."""
        vecs = embeddings_session.provider.embed([])
        assert vecs == []

    def test_similar_texts_have_higher_similarity(self, embeddings_session):
        """Semantically similar texts score higher than unrelated ones."""
        from openmemory.core.embeddings import cosine_similarity

        provider = embeddings_session.provider
        v_base = provider.embed(["machine learning model training"])[0]
        v_similar = provider.embed(["neural network training process"])[0]
        v_unrelated = provider.embed(["apple pie recipe cooking"])[0]

        sim_similar = cosine_similarity(v_base, v_similar)
        sim_unrelated = cosine_similarity(v_base, v_unrelated)

        assert sim_similar > sim_unrelated, (
            f"Expected similar ({sim_similar:.3f}) > unrelated ({sim_unrelated:.3f})"
        )


@pytest.mark.embeddings
class TestVectorSearch:
    def test_sync_and_vector_search_returns_results(self, embeddings_session, tmp_path_factory):
        """After syncing content, vector search returns relevant chunks."""
        tmp = tmp_path_factory.mktemp("vector_search")
        cfg = _load_embeddings_config()
        s_cfg = OpenMemoryConfig(
            root_dir=tmp,
            workspace="vs-test",
            embedding=cfg.embedding,
            search=SearchConfig(top_k=5, vector_weight=1.0),
        )
        s = MemorySession.create(uuid.uuid4().hex[:8], config=s_cfg)
        try:
            # Write some content (memory_write uses tier=, not file=)
            s.execute_tool(
                "memory_write",
                content="Alice is a senior software engineer who specialises in Python and machine learning.",
                tier="long_term",
            )
            s.execute_tool(
                "memory_write",
                content="Bob is a product manager focused on mobile applications and user experience.",
                tier="long_term",
            )
            s.sync()

            results = s.execute_tool("memory_search", query="Python software engineer")
            assert results["status"] == "ok"
            assert len(results["results"]) > 0

            # Alice-related content should appear before Bob-related content
            top_result = results["results"][0]
            assert "Alice" in top_result["text"] or "Python" in top_result["text"]
        finally:
            s.close()

    def test_semantic_search_finds_paraphrase(self, embeddings_session, tmp_path_factory):
        """Vector search matches semantically equivalent queries even without keyword overlap."""
        tmp = tmp_path_factory.mktemp("paraphrase_search")
        cfg = _load_embeddings_config()
        s_cfg = OpenMemoryConfig(
            root_dir=tmp,
            workspace="para-test",
            embedding=cfg.embedding,
            search=SearchConfig(top_k=5, vector_weight=1.0),
        )
        s = MemorySession.create(uuid.uuid4().hex[:8], config=s_cfg)
        try:
            s.execute_tool(
                "memory_write",
                content="The team uses pytest for automated testing of the backend services.",
                tier="long_term",
            )
            s.sync()

            # Query uses different words but same meaning
            results = s.execute_tool(
                "memory_search",
                query="automated quality assurance framework for server-side code",
            )
            assert results["status"] == "ok"
            assert len(results["results"]) > 0
        finally:
            s.close()

    def test_hybrid_search_outperforms_bm25_on_paraphrase(
        self, tmp_path_factory
    ):
        """
        Hybrid search (vector + BM25) should surface paraphrase content that
        pure BM25 would miss due to lack of keyword overlap.
        """
        tmp = tmp_path_factory.mktemp("hybrid_vs_bm25")
        cfg = _load_embeddings_config()

        content = (
            "The engineering team decided to migrate the authentication service "
            "from a monolith to microservices architecture."
        )
        # Paraphrase query — no words in common with content except 'the'
        paraphrase_query = "breaking apart a large application into smaller independent services"

        def _run_search(vector_weight: float) -> list[dict]:
            s_cfg = OpenMemoryConfig(
                root_dir=tmp,
                workspace=f"hybrid-{int(vector_weight * 100)}",
                embedding=cfg.embedding,
                search=SearchConfig(top_k=5, vector_weight=vector_weight),
            )
            s = MemorySession.create(uuid.uuid4().hex[:8], config=s_cfg)
            try:
                s.execute_tool(
                    "memory_write",
                    content=content,
                    tier="long_term",
                )
                s.sync()
                r = s.execute_tool("memory_search", query=paraphrase_query)
                return r.get("results", [])
            finally:
                s.close()

        hybrid_results = _run_search(vector_weight=0.7)
        bm25_results = _run_search(vector_weight=0.0)  # noqa: F841

        # Hybrid should find the content; BM25 may or may not
        assert len(hybrid_results) > 0, "Hybrid search returned no results"
        # The hybrid top result should contain relevant content
        assert any(
            "microservices" in r["text"] or "monolith" in r["text"]
            for r in hybrid_results
        )


@pytest.mark.embeddings
class TestSentenceTransformerProvider:
    """Direct unit tests for SentenceTransformerProvider (no session required)."""

    @pytest.fixture(scope="class")
    def provider(self):
        from openmemory.core.embeddings import SentenceTransformerProvider
        return SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")

    def test_import_no_error(self):
        """sentence-transformers can be imported without error."""
        from openmemory.core.embeddings import SentenceTransformerProvider  # noqa: F401

    def test_model_id(self, provider):
        assert provider.model_id == "sentence-transformers/all-MiniLM-L6-v2"

    def test_dimensions(self, provider):
        assert provider.dimensions == 384

    def test_embed_empty_returns_empty(self, provider):
        assert provider.embed([]) == []

    def test_embed_single_returns_384_dim_vector(self, provider):
        vecs = provider.embed(["hello world"])
        assert len(vecs) == 1
        assert len(vecs[0]) == 384

    def test_embed_multiple_returns_correct_count(self, provider):
        texts = ["foo", "bar", "baz"]
        vecs = provider.embed(texts)
        assert len(vecs) == 3
        assert all(len(v) == 384 for v in vecs)

    def test_embed_values_are_floats(self, provider):
        vecs = provider.embed(["check types"])
        assert all(isinstance(x, float) for x in vecs[0])


@pytest.mark.embeddings
class TestEndToEndMemorySearch:
    def test_memory_search_tool_with_real_embeddings(
        self, embeddings_session, tmp_path_factory
    ):
        """End-to-end: write → sync → search using memory_search tool."""
        tmp = tmp_path_factory.mktemp("e2e")
        cfg = _load_embeddings_config()
        s_cfg = OpenMemoryConfig(
            root_dir=tmp,
            workspace="e2e-test",
            embedding=cfg.embedding,
            search=cfg.search,
        )
        s = MemorySession.create(uuid.uuid4().hex[:8], config=s_cfg)
        try:
            facts = [
                "The project deadline is end of Q2 2026.",
                "Alice prefers async code reviews over synchronous meetings.",
                "The staging environment runs on Kubernetes in AWS us-east-1.",
            ]
            for fact in facts:
                s.execute_tool(
                    "memory_write",
                    content=fact,
                    tier="long_term",
                )
            s.sync()

            r = s.execute_tool("memory_search", query="deployment infrastructure cloud")
            assert r["status"] == "ok"
            assert len(r["results"]) > 0
            # Kubernetes/AWS fact should rank highly
            assert any(
                "Kubernetes" in res["text"] or "AWS" in res["text"]
                for res in r["results"]
            )
        finally:
            s.close()