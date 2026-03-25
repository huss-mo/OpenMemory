"""memory_delete tool - hard-delete lines from a memory file (internal helper).

This module is kept as a standalone internal helper for direct Python API use,
but is no longer registered in ALL_TOOLS. The merged memory_write tool handles
delete mode (content="" + start_line + end_line).
"""
from __future__ import annotations

from groundmemory.tools.base import ok, err, is_immutable, _IMMUTABLE_MSG, sync_after_edit
from groundmemory.core import storage
from groundmemory.core.relations import parse_relations_from_text, _relation_id

SCHEMA = {
    "name": "memory_delete",
    "description": (
        "Hard-delete specific lines from a mutable memory file. "
        "Only USER.md, AGENTS.md, and RELATIONS.md are editable; MEMORY.md and daily/*.md are "
        "append-only history and cannot be modified."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "description": (
                    "Relative path to the mutable file to edit, e.g. 'USER.md', "
                    "'AGENTS.md', or 'RELATIONS.md'. MEMORY.md and daily/*.md are immutable."
                ),
            },
            "start_line": {
                "type": "integer",
                "description": "1-based line number of the first line to delete.",
            },
            "end_line": {
                "type": "integer",
                "description": (
                    "1-based line number of the last line to delete (inclusive). "
                    "Pass the same value as start_line to delete a single line."
                ),
            },
        },
        "required": ["file", "start_line", "end_line"],
    },
}


def run(
    session,
    file: str,
    start_line: int,
    end_line: int,
) -> dict:
    ws = session.workspace

    if is_immutable(file):
        return err(_IMMUTABLE_MSG.format(file=file))

    resolved = ws.resolve_file(file)
    if not resolved.exists():
        return err(f"File not found: {file}")

    is_relations = resolved.name.upper() == "RELATIONS.MD"

    # Snapshot the relation lines that are about to be deleted BEFORE the edit
    relations_to_delete: list[dict] = []
    if is_relations and resolved.exists():
        lines = resolved.read_text(encoding="utf-8").splitlines()
        deleted_text = "\n".join(lines[start_line - 1 : end_line])
        relations_to_delete = parse_relations_from_text(deleted_text)

    try:
        result = storage.hard_delete_lines(resolved, start_line, end_line)
    except (ValueError, IOError) as exc:
        return err(str(exc))

    if "error" in result:
        return err(result["error"])

    # Delete the corresponding SQLite relation rows
    relations_deleted: list[str] = []
    if is_relations and relations_to_delete:
        for r in relations_to_delete:
            rid = _relation_id(r["subject"], r["predicate"], r["object"])
            session.index.delete_relation(rid)
            relations_deleted.append(f"[{r['subject']}] --{r['predicate']}--> [{r['object']}]")

    base_payload = {
        "file": file,
        "deleted_lines": result.get("deleted_lines", f"{start_line}-{end_line}"),
    }
    if relations_deleted:
        base_payload["relations_deleted"] = relations_deleted

    return sync_after_edit(session, resolved, is_relations=False, base_payload=base_payload)