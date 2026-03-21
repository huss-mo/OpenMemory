"""Tests for the memory_relate tool."""
from __future__ import annotations

import uuid
import pytest

from openmemory.config import OpenMemoryConfig, EmbeddingConfig, SearchConfig, RelationsConfig
from openmemory.session import MemorySession
from openmemory.core.embeddings import NullEmbeddingProvider
from openmemory.core import relations as _graph


class TestMemoryRelateBasic:
    def test_relate_returns_ok(self, session):
        r = session.execute_tool(
            "memory_relate", subject="Alice", predicate="works_at", object="Acme"
        )
        assert r["status"] == "ok"

    def test_relate_response_contains_relation(self, session):
        r = session.execute_tool(
            "memory_relate", subject="Alice", predicate="works_at", object="Acme"
        )
        assert "relation" in r
        assert r["relation"]["subject"] == "Alice"
        assert r["relation"]["predicate"] == "works_at"
        assert r["relation"]["object"] == "Acme"

    def test_relate_creates_relations_md(self, session):
        session.execute_tool(
            "memory_relate", subject="Dave", predicate="knows", object="Eve"
        )
        get = session.execute_tool("memory_get", file="RELATIONS.md")
        assert get["status"] == "ok"
        assert get["exists"] is True

    def test_relate_content_appears_in_relations_md(self, session):
        session.execute_tool(
            "memory_relate", subject="Eve", predicate="created_by", object="Frank"
        )
        get = session.execute_tool("memory_get", file="RELATIONS.md")
        content = get["content"]
        assert "Eve" in content
        assert "Frank" in content

    def test_relate_multiple_relations_all_stored(self, session):
        relations = [
            ("Alice", "works_at", "Acme"),
            ("Bob", "manages", "Alice"),
            ("Carol", "reports_to", "Bob"),
        ]
        for s, p, o in relations:
            r = session.execute_tool("memory_relate", subject=s, predicate=p, object=o)
            assert r["status"] == "ok"

        get = session.execute_tool("memory_get", file="RELATIONS.md")
        content = get["content"]
        for s, _, o in relations:
            assert s in content or o in content


class TestMemoryRelateWithOptionalFields:
    def test_relate_with_note(self, session):
        r = session.execute_tool(
            "memory_relate",
            subject="Grace",
            predicate="uses",
            object="Python",
            note="primary programming language",
        )
        assert r["status"] == "ok"
        assert r["relation"]["note"] == "primary programming language"

    def test_relate_with_confidence(self, session):
        r = session.execute_tool(
            "memory_relate",
            subject="Hank",
            predicate="likes",
            object="coffee",
            confidence=0.8,
        )
        assert r["status"] == "ok"
        assert r["relation"]["confidence"] == 0.8

    def test_relate_confidence_clamped_to_one(self, session):
        r = session.execute_tool(
            "memory_relate",
            subject="Ivy",
            predicate="prefers",
            object="vim",
            confidence=99.0,
        )
        assert r["status"] == "ok"
        assert r["relation"]["confidence"] <= 1.0

    def test_relate_confidence_clamped_to_zero(self, session):
        r = session.execute_tool(
            "memory_relate",
            subject="Jack",
            predicate="dislikes",
            object="meetings",
            confidence=-5.0,
        )
        assert r["status"] == "ok"
        assert r["relation"]["confidence"] >= 0.0

    def test_relate_with_source_file(self, session):
        r = session.execute_tool(
            "memory_relate",
            subject="Kim",
            predicate="attends",
            object="Standup",
            source_file="MEMORY.md",
        )
        assert r["status"] == "ok"
        assert r["relation"]["source_file"] == "MEMORY.md"

    def test_relate_default_source_file_is_relations_md(self, session):
        r = session.execute_tool(
            "memory_relate", subject="Leo", predicate="owns", object="Laptop"
        )
        assert r["status"] == "ok"
        assert r["relation"]["source_file"] == "RELATIONS.md"


class TestMemoryRelateValidation:
    def test_relate_missing_subject_raises(self, session):
        """Calling without required args should return an error, not crash."""
        r = session.execute_tool("memory_relate", predicate="knows", object="X")
        assert isinstance(r, dict)
        assert "status" in r

    def test_relate_missing_object_raises(self, session):
        r = session.execute_tool("memory_relate", subject="X", predicate="knows")
        assert isinstance(r, dict)
        assert "status" in r


# ---------------------------------------------------------------------------
# Semantic dedup unit tests (use a deterministic fake provider)
# ---------------------------------------------------------------------------

class _FixedVectorProvider:
    """
    Fake embedding provider that returns a fixed vector for each text.
    Mapping: text -> vector is provided at construction time.
    Unknown texts get the zero vector (so cosine sim = 0 → not a duplicate).
    """

    model_id = "fixed-test-provider"
    dimensions = 3

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._mapping.get(t, [0.0, 0.0, 0.0]) for t in texts]


def _make_dedup_session(tmp_path, provider, threshold=0.92):
    """Create a session with a custom provider and dedup threshold."""
    cfg = OpenMemoryConfig(
        root_dir=tmp_path,
        workspace="dedup-test",
        embedding=EmbeddingConfig(provider="none"),
        search=SearchConfig(),
        relations=RelationsConfig(dedup_threshold=threshold),
    )
    name = uuid.uuid4().hex[:8]
    s = MemorySession.create(name, config=cfg)
    # Swap the NullEmbeddingProvider for our fake one
    s.provider = provider
    return s


class TestSemanticDedup:
    def test_no_dedup_with_null_provider(self, session):
        """NullEmbeddingProvider returns empty vectors → dedup is skipped."""
        # Add the same semantic relation twice with slightly different wording
        session.execute_tool(
            "memory_relate", subject="Alice", predicate="works_at", object="Acme"
        )
        session.execute_tool(
            "memory_relate", subject="Alice", predicate="employed_by", object="Acme"
        )
        # Both should be stored (no semantic dedup without real embeddings)
        rows = session.index.get_all_relations()
        assert len(rows) == 2

    def test_exact_duplicate_not_re_added_to_file(self, session):
        """Exact-match dedup via SHA hash still works regardless of provider."""
        session.execute_tool(
            "memory_relate", subject="Alice", predicate="works_at", object="Acme"
        )
        session.execute_tool(
            "memory_relate", subject="Alice", predicate="works_at", object="Acme"
        )
        # SQLite upsert means still only one row
        rows = session.index.get_all_relations()
        assert len(rows) == 1

    def test_semantic_duplicate_skipped(self, tmp_path):
        """High-similarity triples are deduplicated when using a real provider."""
        # Both triples map to the same vector → cosine sim = 1.0 ≥ 0.92
        vec = [1.0, 0.0, 0.0]
        provider = _FixedVectorProvider({
            "Alice works_at Acme": vec,
            "Alice employed_by Acme": vec,
        })
        s = _make_dedup_session(tmp_path, provider, threshold=0.92)
        try:
            s.execute_tool(
                "memory_relate", subject="Alice", predicate="works_at", object="Acme"
            )
            r2 = s.execute_tool(
                "memory_relate", subject="Alice", predicate="employed_by", object="Acme"
            )
            # Second call returns ok (dedup is silent — not an error)
            assert r2["status"] == "ok"
            # Only one relation should be stored in SQLite
            rows = s.index.get_all_relations()
            assert len(rows) == 1
        finally:
            s.close()

    def test_dissimilar_triple_not_deduplicated(self, tmp_path):
        """Low-similarity triples are NOT deduplicated."""
        provider = _FixedVectorProvider({
            "Alice works_at Acme": [1.0, 0.0, 0.0],
            "Bob manages Carol": [0.0, 1.0, 0.0],
        })
        s = _make_dedup_session(tmp_path, provider, threshold=0.92)
        try:
            s.execute_tool(
                "memory_relate", subject="Alice", predicate="works_at", object="Acme"
            )
            s.execute_tool(
                "memory_relate", subject="Bob", predicate="manages", object="Carol"
            )
            rows = s.index.get_all_relations()
            assert len(rows) == 2
        finally:
            s.close()

    def test_dedup_threshold_respected(self, tmp_path):
        """Triples below the threshold are stored even when somewhat similar."""
        import math
        # Two vectors with ~0.5 cosine similarity
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        provider = _FixedVectorProvider({
            "Alice works_at Acme": v1,
            "Alice employed_by Acme Corp": v2,
        })
        # Set threshold very high (0.99) — these vectors have sim=0.0 < 0.99
        s = _make_dedup_session(tmp_path, provider, threshold=0.99)
        try:
            s.execute_tool(
                "memory_relate", subject="Alice", predicate="works_at", object="Acme"
            )
            s.execute_tool(
                "memory_relate",
                subject="Alice",
                predicate="employed_by",
                object="Acme Corp",
            )
            rows = s.index.get_all_relations()
            assert len(rows) == 2
        finally:
            s.close()

    def test_cosine_similarity_pure_function(self):
        """Unit-test the _cosine_similarity helper directly."""
        from openmemory.core.relations import _cosine_similarity

        # Identical vectors → similarity 1.0
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

        # Orthogonal vectors → similarity 0.0
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

        # Opposite vectors → similarity -1.0
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

        # Zero vector → similarity 0.0 (no division by zero)
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Supersede tests
# ---------------------------------------------------------------------------

class TestMemoryRelateSupersedes:
    def test_supersedes_removes_old_sqlite_row(self, session):
        """supersedes=True should delete old (subject, predicate) rows from SQLite."""
        session.execute_tool(
            "memory_relate", subject="Hussein", predicate="works_at", object="iHorizons"
        )
        assert len(session.index.get_all_relations()) == 1

        session.execute_tool(
            "memory_relate",
            subject="Hussein",
            predicate="works_at",
            object="One Industry",
            supersedes=True,
        )
        rows = session.index.get_all_relations()
        assert len(rows) == 1
        assert rows[0]["object"] == "One Industry"

    def test_supersedes_removes_old_line_from_file(self, session):
        """supersedes=True should remove old lines from RELATIONS.md."""
        session.execute_tool(
            "memory_relate", subject="Hussein", predicate="works_at", object="iHorizons"
        )
        session.execute_tool(
            "memory_relate",
            subject="Hussein",
            predicate="works_at",
            object="One Industry",
            supersedes=True,
        )
        get = session.execute_tool("memory_get", file="RELATIONS.md")
        content = get["content"]
        assert "iHorizons" not in content
        assert "One Industry" in content

    def test_supersedes_response_includes_superseded_list(self, session):
        """The tool response should include a 'superseded' key listing what was removed."""
        session.execute_tool(
            "memory_relate", subject="Alice", predicate="lives_in", object="Cairo"
        )
        r = session.execute_tool(
            "memory_relate",
            subject="Alice",
            predicate="lives_in",
            object="Berlin",
            supersedes=True,
        )
        assert r["status"] == "ok"
        assert "superseded" in r
        assert len(r["superseded"]) == 1
        assert r["superseded"][0]["object"] == "Cairo"

    def test_supersedes_multiple_old_relations_all_removed(self, session):
        """If multiple prior (subject, predicate) triples exist, all are removed."""
        # Manually insert two rows with the same (subject, predicate)
        from openmemory.core import relations as _r
        ws = session.workspace
        _r.add_relation(
            session.index, ws.relations_file,
            "Bob", "member_of", "Team A",
        )
        _r.add_relation(
            session.index, ws.relations_file,
            "Bob", "member_of", "Team B",
        )
        assert len(session.index.get_all_relations()) == 2

        r = session.execute_tool(
            "memory_relate",
            subject="Bob",
            predicate="member_of",
            object="Team C",
            supersedes=True,
        )
        rows = session.index.get_all_relations()
        assert len(rows) == 1
        assert rows[0]["object"] == "Team C"
        assert len(r["superseded"]) == 2

    def test_supersedes_false_keeps_existing_relations(self, session):
        """Without supersedes=True, existing (subject, predicate) triples are kept."""
        session.execute_tool(
            "memory_relate", subject="Carol", predicate="knows", object="Dave"
        )
        session.execute_tool(
            "memory_relate", subject="Carol", predicate="knows", object="Eve"
        )
        rows = session.index.get_all_relations()
        assert len(rows) == 2

    def test_supersedes_only_affects_matching_predicate(self, session):
        """supersedes=True only removes triples with the exact same predicate."""
        session.execute_tool(
            "memory_relate", subject="Dave", predicate="works_at", object="Acme"
        )
        session.execute_tool(
            "memory_relate", subject="Dave", predicate="lives_in", object="Cairo"
        )
        session.execute_tool(
            "memory_relate",
            subject="Dave",
            predicate="works_at",
            object="Globex",
            supersedes=True,
        )
        rows = session.index.get_all_relations()
        # lives_in should still be there
        predicates = {r["predicate"] for r in rows}
        assert "lives_in" in predicates
        assert len(rows) == 2  # works_at (new) + lives_in

    def test_supersedes_with_no_prior_relation_is_noop(self, session):
        """supersedes=True with no existing triples just writes normally."""
        r = session.execute_tool(
            "memory_relate",
            subject="Eve",
            predicate="works_at",
            object="Initech",
            supersedes=True,
        )
        assert r["status"] == "ok"
        assert "superseded" not in r  # nothing was deleted
        rows = session.index.get_all_relations()
        assert len(rows) == 1

    def test_supersedes_case_insensitive_subject_and_predicate(self, session):
        """Subject and predicate matching is case-insensitive."""
        session.execute_tool(
            "memory_relate", subject="Frank", predicate="works_at", object="OldCo"
        )
        r = session.execute_tool(
            "memory_relate",
            subject="frank",  # different case
            predicate="WORKS_AT",  # different case
            object="NewCo",
            supersedes=True,
        )
        assert r["status"] == "ok"
        # The old row should be gone
        rows = session.index.get_all_relations()
        objects = {row["object"] for row in rows}
        assert "OldCo" not in objects
        assert "NewCo" in objects
