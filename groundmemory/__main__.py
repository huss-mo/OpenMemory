"""
groundmemory CLI entry point.

Usage
-----
Restore a workspace backup:

    groundmemory --restore -1                     # most recent backup
    groundmemory --restore 2026-04-08             # exact date (error if ambiguous)
    groundmemory --restore 2026-04-08_165530      # exact timestamp

List available backups:

    groundmemory --list-backups
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _get_workspace_path() -> Path:
    """Resolve the active workspace path from the environment / defaults."""
    from groundmemory.config import groundmemoryConfig
    cfg = groundmemoryConfig.auto()
    return cfg.root_dir / cfg.workspace


def cmd_list_backups(workspace_path: Path) -> None:
    from groundmemory.core.backup import list_backups
    backups = list_backups(workspace_path)
    if not backups:
        print(f"No backups found for workspace: {workspace_path}")
        return
    print(f"Backups for workspace '{workspace_path.name}':")
    for i, b in enumerate(reversed(backups), start=1):
        print(f"  -{i}  {b.stem}")


def cmd_restore(spec: str, workspace_path: Path) -> None:
    from groundmemory.core.backup import list_backups, parse_spec, restore_backup

    backups = list_backups(workspace_path)
    if not backups:
        print(f"No backups found for workspace: {workspace_path}", file=sys.stderr)
        sys.exit(1)

    target = parse_spec(spec, backups)

    if target is None:
        # Check for ambiguous date match
        matches = [b for b in backups if b.stem.startswith(spec)]
        if matches:
            print(
                f"Ambiguous spec '{spec}' matches {len(matches)} backups. "
                "Specify an exact timestamp:\n",
                file=sys.stderr,
            )
            for m in matches:
                print(f"  {m.stem}", file=sys.stderr)
        else:
            print(f"No backup found matching '{spec}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Restoring backup: {target.stem}")
    print(f"  → workspace: {workspace_path}")
    print()
    answer = input("Proceed? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        print("Aborted.")
        sys.exit(0)

    restore_backup(target, workspace_path)
    print(f"\nRestored successfully from '{target.stem}'.")
    print("If an MCP server is running against this workspace, restart it.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="groundmemory",
        description="GroundMemory workspace management CLI.",
    )
    parser.add_argument(
        "--restore",
        metavar="SPEC",
        help=(
            "Restore a workspace backup. SPEC can be: "
            "'-1' (most recent), '-2' (second-most-recent), "
            "'YYYY-MM-DD' (exact date), or 'YYYY-MM-DD_HHmmss' (exact timestamp)."
        ),
    )
    parser.add_argument(
        "--list-backups",
        action="store_true",
        help="List all available backups for the current workspace.",
    )
    args = parser.parse_args()

    workspace_path = _get_workspace_path()

    if args.list_backups:
        cmd_list_backups(workspace_path)
    elif args.restore:
        cmd_restore(args.restore, workspace_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()