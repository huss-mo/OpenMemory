"""memory_relate tool — record a named relationship between two entities."""
from __future__ import annotations

from openmemory.tools.base import ok, err
from openmemory.core import graph as _graph

SCHEMA = {
    "name": "memory_relate",
    "description": (
        "Record a directional relationship between two entities so the memory "
        "graph can surface relevant connections during searches. "
        "Example: subject='Alice', predicate='works_at', object='Acme Corp'."
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
                    "Examples: 'works_at', 'knows', 'part_of', 'created_by'."
                ),
            },
            "object": {
                "type": "string",
                "description": "The entity that is the target of the relationship.",
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
    note: str = "",
    source_file: str = "RELATIONS.md",
    confidence: float = 1.0,
) -> dict:
    ws = session.workspace

    try:
        _graph.add_relation(
            index=session.index,
            workspace=ws,
            subject=subject.strip(),
            predicate=predicate.strip(),
            obj=object.strip(),
            note=note,
            source_file=source_file,
            confidence=max(0.0, min(1.0, confidence)),
        )
    except Exception as exc:  # noqa: BLE001
        return err(f"Failed to record relation: {exc}")

    return ok(
        {
            "relation": {
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "note": note,
                "source_file": source_file,
                "confidence": confidence,
            }
        }
    )