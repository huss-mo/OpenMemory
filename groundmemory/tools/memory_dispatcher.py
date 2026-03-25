"""
memory_tool - single-tool dispatcher for maximum token efficiency.

When dispatcher_mode is enabled, this single tool replaces all individual
memory tools. The model passes an `action` one-liner and the dispatcher
routes to the correct implementation.

Special action: "describe:<tool_name>" returns the full JSON schema for that
tool so the model can inspect parameter details on demand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from groundmemory.tools.base import err, ok

if TYPE_CHECKING:
    from groundmemory.session import MemorySession

# One-liner descriptions for each action (shown in the dispatcher schema).
# "bootstrap" is handled inline (not delegated to the standalone memory_bootstrap
# module) because in dispatcher_mode the standalone tool is not registered - the
# dispatcher is the only registered tool and must handle all operations itself.
_ACTION_DOCS = {
    "bootstrap":  "Return the full memory bootstrap context (no args needed). Must be called **once at the very start of every session** before doing anything else.",
    "describe":   "Return full schema for an action (args: {\"action\": \"<name>\"}). Call this once before invoking action (other than bootstrap) to understand what args it needs.",
    "read":       "Search memory or read a file. Args: query?, file?, top_k?, start_line?, end_line?",
    "write":      "Append/replace/delete memory. Args: file, content, search?, start_line?, end_line?, tags?",
    "relate":     "Add an entity relation. Args: subject, predicate, object, note?",
    "list":       "List all memory files with sizes. No args.",
}

SCHEMA = {
    "name": "memory_tool",
    "description": (
        "Unified memory dispatcher. Pass `action` to select the operation:\n"
        + "\n".join(f"  {k}: {v}" for k, v in _ACTION_DOCS.items())
        + "\n\nUse action='describe' with the target action name in `args.action` "
        "to retrieve the full parameter schema for that operation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_ACTION_DOCS.keys()),
                "description": "Which memory operation to perform.",
            },
            "args": {
                "type": "object",
                "description": "Arguments for the selected action (see action descriptions above).",
                "additionalProperties": True,
            },
        },
        "required": ["action"],
    },
}

# Mapping from action name → (module, runner_attr)
# We import lazily inside run() to avoid circular imports.
_ACTION_TO_MODULE = {
    "read":      ("groundmemory.tools.memory_read",      "run"),
    "write":     ("groundmemory.tools.memory_write",     "run"),
    "bootstrap": (None,                                   None),   # handled inline
    "relate":    ("groundmemory.tools.memory_relate",    "run"),
    "list":      ("groundmemory.tools.memory_list",      "run"),
    "describe":  (None,                                   None),   # handled inline
}

# Full schemas for describe action (lazy-imported tool schemas by action name)
_ACTION_TO_SCHEMA_MODULE = {
    "read":      ("groundmemory.tools.memory_read",    "SCHEMA"),
    "write":     ("groundmemory.tools.memory_write",   "SCHEMA"),
    "relate":    ("groundmemory.tools.memory_relate",  "SCHEMA"),
    "list":      ("groundmemory.tools.memory_list",    "SCHEMA"),
    "bootstrap": ("groundmemory.tools.memory_bootstrap", "SCHEMA"),
}


def run(
    session: "MemorySession",
    action: str,
    args: Optional[dict[str, Any]] = None,
) -> dict:
    if args is None:
        args = {}

    # ------------------------------------------------------------------
    # describe - return full schema for the requested action
    # ------------------------------------------------------------------
    if action == "describe":
        target = args.get("action", "")
        if not target:
            return err("Provide args.action with the name of the action to describe.")
        if target not in _ACTION_TO_SCHEMA_MODULE:
            return err(
                f"No schema available for action '{target}'. "
                f"Available: {list(_ACTION_TO_SCHEMA_MODULE.keys())}"
            )
        mod_name, attr = _ACTION_TO_SCHEMA_MODULE[target]
        import importlib
        mod = importlib.import_module(mod_name)
        schema = getattr(mod, attr)
        return ok({"action": target, "schema": schema})

    # ------------------------------------------------------------------
    # bootstrap - inline handler
    # ------------------------------------------------------------------
    if action == "bootstrap":
        try:
            prompt = session.bootstrap()
        except Exception as exc:
            return err(f"bootstrap failed: {exc}")
        return ok({"bootstrap": prompt})

    # ------------------------------------------------------------------
    # Delegated actions
    # ------------------------------------------------------------------
    mod_name, runner_attr = _ACTION_TO_MODULE.get(action, (None, None))
    if mod_name is None:
        return err(
            f"Unknown action '{action}'. "
            f"Valid actions: {list(_ACTION_DOCS.keys())}"
        )

    import importlib
    mod = importlib.import_module(mod_name)
    runner = getattr(mod, runner_attr)

    try:
        return runner(session, **args)
    except TypeError as exc:
        return err(f"Invalid args for action '{action}': {exc}")
    except Exception as exc:
        return err(f"Action '{action}' failed: {exc}")