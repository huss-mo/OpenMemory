"""
memory_write tool - unified append / replace / hard-delete for memory files.

Dispatch logic (in priority order):
  1. `search` provided                              → replace_text (first-occurrence text swap)
  2. `start_line` + `end_line` + `content == ""`   → hard-delete lines (physical removal)
  3. `start_line` + `end_line` + `content` non-empty → replace_lines
  4. else                                           → append to the file named by `file`

Append filename → tier mapping:
  MEMORY.md           → long_term  (append-only, timestamped)
  USER.md             → user       (append-only, timestamped)
  AGENTS.md           → agent      (append-only, timestamped)
  daily               → daily      (today's log)
  daily/YYYY-MM-DD.md → rejected   (daily logs are immutable once written)

Edit operations (replace / delete) are allowed on:
  USER.md, AGENTS.md, RELATIONS.md
Edit operations are NOT allowed on MEMORY.md or daily/*.md (append-only).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from groundmemory.tools.base import err, ok, is_immutable, _IMMUTABLE_MSG, sync_after_edit

if TYPE_CHECKING:
    from groundmemory.session import MemorySession

# Filenames that accept append writes and which tier they map to
_APPEND_FILE_TO_TIER: dict[str, str] = {
    "MEMORY.md": "long_term",
    "USER.md": "user",
    "AGENTS.md": "agent",
    "daily": "daily",
}

# Relations format reminder (imported lazily to avoid circular imports at parse time)
_RELATIONS_FORMAT_REMINDER = (
    "Each non-blank line must follow: - [Subject] --predicate--> [Object] (YYYY-MM-DD)"
)

SCHEMA = {
    "name": "memory_write",
    "description": (
        "Write to memory. Four modes selected by the parameters you supply:\n\n"
        "  APPEND       - omit start_line/end_line/search; `file` + `content` appends to that file.\n"
        "  REPLACE_TEXT - supply `search`; replaces the first exact occurrence with `content`.\n"
        "  REPLACE_LINES- supply `start_line` + `end_line` + non-empty `content`; replaces that line range.\n"
        "  DELETE       - supply `start_line` + `end_line` + `content=\"\"`; hard-deletes those lines.\n\n"
        "Append targets: 'MEMORY.md' (long-term facts), 'USER.md' (user profile), "
        "'AGENTS.md' (agent rules), 'daily' (today's log).\n"
        "Edit targets (replace/delete): 'USER.md', 'AGENTS.md', 'RELATIONS.md'.\n"
        "MEMORY.md and daily/*.md are append-only and cannot be edited or deleted.\n\n"
        "Before appending to MEMORY.md, USER.md, or AGENTS.md use memory_read with a query "
        "to check for near-duplicates; prefer REPLACE_TEXT or REPLACE_LINES to update existing entries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "description": (
                    "Target file. Append targets: 'MEMORY.md', 'USER.md', 'AGENTS.md', 'daily'. "
                    "Edit targets: 'USER.md', 'AGENTS.md', 'RELATIONS.md'."
                ),
            },
            "content": {
                "type": "string",
                "description": (
                    "Content to write or the replacement text. "
                    "Pass empty string (\"\") together with start_line+end_line to hard-delete those lines."
                ),
            },
            "search": {
                "type": "string",
                "description": (
                    "REPLACE_TEXT mode: exact string to find in the file (first occurrence replaced). "
                    "Must match character-for-character including whitespace."
                ),
            },
            "start_line": {
                "type": "integer",
                "description": "1-based line number of the first line to replace or delete.",
                "minimum": 1,
            },
            "end_line": {
                "type": "integer",
                "description": "1-based line number of the last line to replace or delete (inclusive).",
                "minimum": 1,
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for APPEND to MEMORY.md or daily (e.g. ['preference', 'project-x']).",
            },
        },
        "required": ["file", "content"],
    },
}


def run(
    session: "MemorySession",
    file: str = "MEMORY.md",
    content: str = "",
    search: Optional[str] = None,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    from groundmemory.core import storage
    from groundmemory.core.relations import (
        validate_relations_replacement,
        RELATIONS_FORMAT_REMINDER,
    )

    # -----------------------------------------------------------------------
    # Mode 1: REPLACE_TEXT  (search param provided)
    # -----------------------------------------------------------------------
    if search is not None:
        if is_immutable(file):
            return err(_IMMUTABLE_MSG.format(file=file))
        if not search.strip():
            return err("'search' cannot be empty or whitespace.")

        resolved = session.workspace.resolve_file(file)
        if not resolved.exists():
            return err(f"File not found: {file}")

        is_relations = resolved.name.upper() == "RELATIONS.MD"
        if is_relations:
            ok_flag, _, invalid = validate_relations_replacement(content)
            if not ok_flag:
                return err(
                    f"Replacement contains {len(invalid)} invalid line(s): {invalid}\n"
                    f"{RELATIONS_FORMAT_REMINDER}"
                )

        result = storage.replace_text(resolved, search, content)
        if "error" in result:
            return err(result["error"])

        return sync_after_edit(
            session,
            resolved,
            is_relations,
            {"file": file, "mode": "replace_text", "replaced": True, "chars_delta": result.get("chars_delta", 0)},
        )

    # -----------------------------------------------------------------------
    # Mode 2 & 3: line-range operations (start_line + end_line provided)
    # -----------------------------------------------------------------------
    if start_line is not None and end_line is not None:
        if is_immutable(file):
            return err(_IMMUTABLE_MSG.format(file=file))

        resolved = session.workspace.resolve_file(file)
        if not resolved.exists():
            return err(f"File not found: {file}")

        is_relations = resolved.name.upper() == "RELATIONS.MD"

        # Mode 2: DELETE (content == "")
        if content == "":
            from groundmemory.core.relations import parse_relations_from_text, _relation_id

            # Snapshot relations to be removed before touching the file
            relations_to_delete: list[dict] = []
            if is_relations:
                lines = resolved.read_text(encoding="utf-8").splitlines()
                deleted_text = "\n".join(lines[start_line - 1 : end_line])
                relations_to_delete = parse_relations_from_text(deleted_text)

            result = storage.hard_delete_lines(resolved, start_line, end_line)
            if "error" in result:
                return err(result["error"])

            # Remove deleted relation rows from SQLite
            relations_deleted: list[str] = []
            if is_relations and relations_to_delete:
                for r in relations_to_delete:
                    rid = _relation_id(r["subject"], r["predicate"], r["object"])
                    session.index.delete_relation(rid)
                    relations_deleted.append(f"[{r['subject']}] --{r['predicate']}--> [{r['object']}]")

            payload: dict = {
                "file": file,
                "mode": "delete",
                "deleted_lines": result.get("deleted_lines", f"{start_line}-{end_line}"),
            }
            if relations_deleted:
                payload["relations_deleted"] = relations_deleted

            return sync_after_edit(session, resolved, is_relations=False, base_payload=payload)

        # Mode 3: REPLACE_LINES (content non-empty)
        if is_relations:
            ok_flag, _, invalid = validate_relations_replacement(content)
            if not ok_flag:
                return err(
                    f"Replacement contains {len(invalid)} invalid line(s): {invalid}\n"
                    f"{RELATIONS_FORMAT_REMINDER}"
                )

        result = storage.replace_lines(resolved, start_line, end_line, content)
        if "error" in result:
            return err(result["error"])

        return sync_after_edit(
            session,
            resolved,
            is_relations,
            {
                "file": file,
                "mode": "replace_lines",
                "replaced_lines": result.get("replaced_lines", f"{start_line}-{end_line}"),
                "replaced_preview": result.get("replaced_preview", ""),
                "chars_delta": result.get("chars_delta", 0),
            },
        )

    # -----------------------------------------------------------------------
    # Mode 4: APPEND
    # -----------------------------------------------------------------------
    if not content or not content.strip():
        return err("'content' cannot be empty for append mode.")

    file_key = file.strip()
    file_upper = file_key.upper()

    # Reject daily/YYYY-MM-DD.md - those are immutable once written
    if file_upper.startswith("DAILY/") or file_upper.startswith("DAILY\\"):
        return err(
            f"'{file}' is a specific daily log and is immutable. "
            "Use 'daily' to append to today's log."
        )

    # Determine tier from filename
    tier: Optional[str] = None
    for fname, t in _APPEND_FILE_TO_TIER.items():
        if file_upper == fname.upper():
            tier = t
            break

    if tier is None:
        return err(
            f"Unknown append target '{file}'. "
            f"Valid targets: {list(_APPEND_FILE_TO_TIER.keys())}."
        )

    from groundmemory.core import sync

    body = content.strip()

    if tier == "long_term":
        if tags:
            body = " ".join(f"#{t}" for t in tags) + "\n" + body
        result = storage.write_long_term(session.workspace, body)
        sync.sync_file(session.workspace.memory_file, session.index, session.provider, session.config.chunking)

    elif tier == "user":
        result = storage.write_user(session.workspace, body)
        sync.sync_file(session.workspace.user_file, session.index, session.provider, session.config.chunking)

    elif tier == "agent":
        result = storage.write_agents(session.workspace, body)
        sync.sync_file(session.workspace.agents_file, session.index, session.provider, session.config.chunking)

    else:  # daily
        if tags:
            body = " ".join(f"#{t}" for t in tags) + "\n" + body
        result = storage.write_daily(session.workspace, body)
        daily_path = session.workspace.daily_file()
        sync.sync_file(daily_path, session.index, session.provider, session.config.chunking)

    result["mode"] = "append"
    return ok(result)