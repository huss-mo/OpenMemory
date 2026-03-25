"""memory_replace_text and memory_replace_lines tools - in-place edits to memory files."""
from __future__ import annotations

from groundmemory.tools.base import ok, err, is_immutable, _IMMUTABLE_MSG, sync_after_edit
from groundmemory.core import storage
from groundmemory.core.relations import (
    validate_relations_replacement,
    RELATIONS_FORMAT_REMINDER,
)

# Back-compat aliases (tests may import the private names directly)
_validate_relations_replacement = validate_relations_replacement
_RELATIONS_FORMAT_REMINDER = RELATIONS_FORMAT_REMINDER


# ---------------------------------------------------------------------------
# memory_replace_text
# ---------------------------------------------------------------------------

SCHEMA_TEXT = {
    "name": "memory_replace_text",
    "description": (
        "Replace the first occurrence of an exact string in a mutable memory file with new text. "
        "Use this to correct or update a specific passage without rewriting the whole file. "
        "The search string must match the file content exactly (including whitespace). "
        "Use memory_get first to read the file and confirm the exact text to replace. "
        "Only USER.md, AGENTS.md, and RELATIONS.md are editable; MEMORY.md and daily/*.md are "
        "append-only history and cannot be modified. "
        "When editing RELATIONS.md every replacement line must follow the format: "
        "- [Subject] --predicate--> [Object] (YYYY-MM-DD)"
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
            "search": {
                "type": "string",
                "description": (
                    "Exact string to search for in the file. Must match character-for-character "
                    "including whitespace and newlines. Only the first occurrence is replaced."
                ),
            },
            "replacement": {
                "type": "string",
                "description": (
                    "Text to substitute in place of the matched string. "
                    "When editing RELATIONS.md, each non-blank line must follow: "
                    "- [Subject] --predicate--> [Object] (YYYY-MM-DD)"
                ),
            },
        },
        "required": ["file", "search", "replacement"],
    },
}


def run_text(session, file: str, search: str, replacement: str) -> dict:
    if is_immutable(file):
        return err(_IMMUTABLE_MSG.format(file=file))

    ws = session.workspace
    resolved = ws.resolve_file(file)

    if not search or not search.strip():
        return err("search cannot be empty")

    is_relations = resolved.name.upper() == "RELATIONS.MD"

    # Validate RELATIONS.md replacement format before touching the file
    if is_relations:
        all_valid, valid_lines, invalid_lines = _validate_relations_replacement(replacement)
        if not all_valid:
            return err(
                f"Replacement text contains {len(invalid_lines)} line(s) that do not match "
                f"the required RELATIONS.md format.\n"
                f"Invalid line(s): {invalid_lines}\n"
                f"{_RELATIONS_FORMAT_REMINDER}"
            )

    result = storage.replace_text(resolved, search, replacement)

    if "error" in result:
        return err(result["error"])

    return sync_after_edit(
        session,
        resolved,
        is_relations,
        {"file": file, "replaced": True, "chars_delta": result.get("chars_delta", 0)},
    )


# ---------------------------------------------------------------------------
# memory_replace_lines
# ---------------------------------------------------------------------------

SCHEMA_LINES = {
    "name": "memory_replace_lines",
    "description": (
        "Replace a range of lines in a mutable memory file with new text. "
        "Use this when you know the line numbers of the content to update. "
        "Call memory_get first to read the file and identify the target line numbers. "
        "Only USER.md, AGENTS.md, and RELATIONS.md are editable; MEMORY.md and daily/*.md are "
        "append-only history and cannot be modified. "
        "When editing RELATIONS.md every replacement line must follow the format: "
        "- [Subject] --predicate--> [Object] (YYYY-MM-DD)"
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
                "description": "1-based line number of the first line to replace.",
            },
            "end_line": {
                "type": "integer",
                "description": (
                    "1-based line number of the last line to replace (inclusive). "
                    "Pass the same value as start_line to replace a single line."
                ),
            },
            "replacement": {
                "type": "string",
                "description": (
                    "Text that replaces the specified line range. "
                    "When editing RELATIONS.md, each non-blank line must follow: "
                    "- [Subject] --predicate--> [Object] (YYYY-MM-DD)"
                ),
            },
        },
        "required": ["file", "start_line", "end_line", "replacement"],
    },
}


def run_lines(
    session,
    file: str,
    start_line: int,
    end_line: int,
    replacement: str,
) -> dict:
    if is_immutable(file):
        return err(_IMMUTABLE_MSG.format(file=file))

    ws = session.workspace
    resolved = ws.resolve_file(file)

    if not resolved.exists():
        return err(f"File not found: {file}")

    is_relations = resolved.name.upper() == "RELATIONS.MD"

    # Validate RELATIONS.md replacement format before touching the file
    if is_relations:
        all_valid, valid_lines, invalid_lines = _validate_relations_replacement(replacement)
        if not all_valid:
            return err(
                f"Replacement text contains {len(invalid_lines)} line(s) that do not match "
                f"the required RELATIONS.md format.\n"
                f"Invalid line(s): {invalid_lines}\n"
                f"{_RELATIONS_FORMAT_REMINDER}"
            )

    result = storage.replace_lines(resolved, start_line, end_line, replacement)

    if "error" in result:
        return err(result["error"])

    return sync_after_edit(
        session,
        resolved,
        is_relations,
        {
            "file": file,
            "replaced_lines": result.get("replaced_lines", f"{start_line}-{end_line}"),
            "replaced_preview": result.get("replaced_preview", ""),
            "chars_delta": result.get("chars_delta", 0),
        },
    )
