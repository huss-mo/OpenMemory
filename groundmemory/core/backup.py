"""
Workspace backup utilities.

Creates timestamped zip archives of the workspace before compaction so the
user can restore a previous state if the agent makes a mistake.

Backup location: <workspace>/backups/YYYY-MM-DD_HHmmss.zip

Archive contents:
  - MEMORY.md, USER.md, AGENTS.md, RELATIONS.md (workspace root)
  - FIRST_RUN.md (workspace root, if present)
  - daily/           (all daily log files)
  - .index/memory.db (SQLite index)
"""
from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

BACKUP_DIR_NAME = "backups"
TIMESTAMP_FORMAT = "%Y-%m-%d_%H%M%S"
DATE_FORMAT = "%Y-%m-%d"


def backup_dir(workspace_path: Path) -> Path:
    """Return (and create if needed) the backups directory for *workspace_path*."""
    d = workspace_path / BACKUP_DIR_NAME
    d.mkdir(exist_ok=True)
    return d


def create_backup(workspace_path: Path) -> Path:
    """
    Zip the workspace into a timestamped archive and return the archive path.

    The backup captures:
    - All Markdown files in the workspace root
    - The entire ``daily/`` sub-directory
    - ``.index/memory.db``

    Raises
    ------
    OSError
        If the backup directory cannot be created or the zip cannot be written.
    """
    stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    archive_path = backup_dir(workspace_path) / f"{stamp}.zip"

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Markdown files in root
        for md_file in sorted(workspace_path.glob("*.md")):
            zf.write(md_file, arcname=md_file.name)

        # Daily logs
        daily_dir = workspace_path / "daily"
        if daily_dir.is_dir():
            for daily_file in sorted(daily_dir.rglob("*")):
                if daily_file.is_file():
                    zf.write(daily_file, arcname=str(daily_file.relative_to(workspace_path)))

        # SQLite index
        db_path = workspace_path / ".index" / "memory.db"
        if db_path.exists():
            zf.write(db_path, arcname=str(db_path.relative_to(workspace_path)))

    return archive_path


# ---------------------------------------------------------------------------
# Restore helpers
# ---------------------------------------------------------------------------

def list_backups(workspace_path: Path) -> list[Path]:
    """Return all backup zip files sorted oldest-first."""
    d = backup_dir(workspace_path)
    return sorted(d.glob("*.zip"))


def parse_spec(spec: str, backups: list[Path]) -> Path | None:
    """
    Resolve a user-supplied restore spec to a backup path.

    Accepted formats
    ----------------
    ``-1``                  most recent backup
    ``-2``                  second-most-recent backup, etc.
    ``YYYY-MM-DD``          exact date; returns None if ambiguous (multiple)
    ``YYYY-MM-DD_HHmmss``   exact timestamp match
    """
    if not backups:
        return None

    # Relative index: -1, -2, ...
    if spec.lstrip("-").isdigit() and spec.startswith("-"):
        idx = int(spec)  # e.g. -1 → last element
        try:
            return backups[idx]
        except IndexError:
            return None

    # Exact timestamp match
    for b in backups:
        if b.stem == spec:
            return b

    # Date-only match
    matches = [b for b in backups if b.stem.startswith(spec)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Ambiguous — caller should print the list and exit
        return None

    return None


def restore_backup(archive_path: Path, workspace_path: Path) -> None:
    """
    Extract *archive_path* into *workspace_path*, overwriting existing files.

    The ``.index/`` directory is created if it does not exist before extraction.
    """
    (workspace_path / ".index").mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(workspace_path)