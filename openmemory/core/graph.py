"""
Relation graph management.

Stores named relationships between entities in:
  1. SQLite (fast lookup, structured queries)
  2. RELATIONS.md (human-readable mirror, injected at bootstrap)

Format in RELATIONS.md:
  - [Alice] --leads--> [Auth Team] (2026-03-20) — "Added during sprint planning"
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from openmemory.core.index import MemoryIndex
from openmemory.core.storage import _atomic_write


def _relation_id(subject: str, predicate: str, object_: str) -> str:
    """Deterministic ID so duplicate relations get upserted, not duplicated."""
    raw = f"{subject.lower()}|{predicate.lower()}|{object_.lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _format_relation_line(
    subject: str,
    predicate: str,
    object_: str,
    note: Optional[str] = None,
) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    line = f"- [{subject}] --{predicate}--> [{object_}] ({date_str})"
    if note:
        line += f' — "{note}"'
    return line


def add_relation(
    index: MemoryIndex,
    relations_file: Path,
    subject: str,
    predicate: str,
    object_: str,
    note: Optional[str] = None,
    confidence: float = 1.0,
) -> dict:
    """
    Record a named relationship between two entities.

    - Writes to SQLite relations table (upsert by deterministic ID).
    - Appends a human-readable line to RELATIONS.md.

    Returns a dict describing what was written.
    """
    subject = subject.strip()
    predicate = predicate.strip()
    object_ = object_.strip()

    relation_id = _relation_id(subject, predicate, object_)

    # 1. Upsert into SQLite
    index.insert_relation(
        relation_id=relation_id,
        subject=subject,
        predicate=predicate,
        object_=object_,
        note=note,
        source_file=str(relations_file),
        confidence=confidence,
    )

    # 2. Append to RELATIONS.md (idempotent-ish: check if line already present)
    line = _format_relation_line(subject, predicate, object_, note)
    existing = relations_file.read_text(encoding="utf-8") if relations_file.exists() else ""

    # Only append if this exact triple isn't already in the file
    marker = f"[{subject}] --{predicate}--> [{object_}]"
    if marker not in existing:
        new_content = existing.rstrip() + "\n" + line + "\n"
        _atomic_write(relations_file, new_content)

    return {
        "id": relation_id,
        "subject": subject,
        "predicate": predicate,
        "object": object_,
        "note": note,
        "written_to": str(relations_file),
    }


def get_relations(index: MemoryIndex, entity: Optional[str] = None) -> list[dict]:
    """
    Return relations as a list of dicts.
    If *entity* is given, filter to relations involving that entity.
    """
    if entity:
        rows = index.get_relations_for_entity(entity)
    else:
        rows = index.get_all_relations()

    return [
        {
            "id": row["id"],
            "subject": row["subject"],
            "predicate": row["predicate"],
            "object": row["object"],
            "note": row["note"],
            "source_file": row["source_file"],
            "created_at": datetime.fromtimestamp(row["created_at"]).isoformat(),
            "confidence": row["confidence"],
        }
        for row in rows
    ]


def format_relations_for_context(relations: list[dict]) -> str:
    """Format a list of relation dicts as a compact Markdown block for injection."""
    if not relations:
        return ""
    lines = ["## Relations\n"]
    for r in relations:
        line = f"- [{r['subject']}] --{r['predicate']}--> [{r['object']}]"
        if r.get("note"):
            line += f" — {r['note']}"
        lines.append(line)
    return "\n".join(lines)