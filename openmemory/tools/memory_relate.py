"""memory_relate tool — record a named relationship between two entities."""
from __future__ import annotations

from openmemory.tools.base import ok, err
from openmemory.core import relations as _relations

SCHEMA = {
    "name": "memory_relate",
    "description": (
        "Record a directional relationship between two entities so the memory "
        "graph can surface relevant connections during searches. "
        "Example: subject='Alice', predicate='works_at', object='Acme Corp'.\n\n"
        "Before calling this tool, use memory_search to check whether a conflicting "
        "relation already exists.\n\n"
        "Set supersedes=True when the new relation REPLACES a prior one for the same "
        "subject+predicate — for example, when someone changes jobs, moves cities, or "
        "changes teams. This removes all prior (subject, predicate) triples before "
        "writing the new one. Do NOT set supersedes=True when multiple objects can be "
        "valid simultaneously (e.g. a person can attend multiple meetups)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "The entity that is the source of the relationship.",
            },
            "predicate": {
                "type": "string",
                "description": (
                    "The relationship type, preferably snake_case. "
                    "Examples: 'works_at', 'lives_in', 'knows', 'part_of', 'created_by'."
                ),
            },
            "object": {
                "type": "string",
                "description": "The entity that is the target of the relationship.",
            },
            "supersedes": {
                "type": "boolean",
                "description": (
                    "Set to true when this relation REPLACES all prior relations with the "
                    "same subject and predicate (e.g. job change, location change, team "
                    "reassignment). All existing (subject, predicate) triples will be "
                    "deleted before the new one is written. Default false."
                ),
                "default": False,
            },
            "note": {
                "type": "string",
                "description": "Optional free-text annotation about this relationship.",
                "default": "",
            },
            "source_file": {
                "type": "string",
                "description": (
                    "Optional. The memory file this relation was derived from "
                    "(e.g. 'daily/2025-01-01.md'). Defaults to 'RELATIONS.md'."
                ),
                "default": "RELATIONS.md",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score between 0.0 and 1.0 (default 1.0).",
                "default": 1.0,
            },
        },
        "required": ["subject", "predicate", "object"],
    },
}


def run(
    session,
    subject: str,
    predicate: str,
    object: str,  # noqa: A002  (shadow builtin intentionally for schema clarity)
    supersedes: bool = False,
    note: str = "",
    source_file: str = "RELATIONS.md",
    confidence: float = 1.0,
) -> dict:
    ws = session.workspace

    clamped_confidence = max(0.0, min(1.0, confidence))

    try:
        result = _relations.add_relation(
            index=session.index,
            relations_file=ws.relations_file,
            subject=subject.strip(),
            predicate=predicate.strip(),
            object_=object.strip(),
            note=note,
            confidence=clamped_confidence,
            provider=session.provider,
            dedup_threshold=session.config.relations.dedup_threshold,
            supersedes=supersedes,
        )
    except Exception as exc:  # noqa: BLE001
        return err(f"Failed to record relation: {exc}")

    payload: dict = {
        "relation": {
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "note": note,
            "source_file": source_file,
            "confidence": clamped_confidence,
        }
    }
    if result.get("deduplicated"):
        payload["deduplicated"] = True
        payload["duplicate_of"] = result["duplicate_of"]
    if result.get("superseded"):
        payload["superseded"] = result["superseded"]
    return ok(payload)
