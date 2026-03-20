"""Tests for the memory_relate tool."""
from __future__ import annotations


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