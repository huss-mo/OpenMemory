"""
Workspace management — directory layout, initialization, and file path resolution.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

# Default content for workspace files created on first init
_DEFAULT_MEMORY_MD = """\
# Memory

Curated long-term knowledge: preferences, key decisions, persistent facts.
The agent writes here when something important should be remembered across sessions.
"""

_DEFAULT_USER_MD = """\
# User Profile

Add stable facts about the user here — name, role, working style, preferences.
"""

_DEFAULT_AGENTS_MD = """\
# Agent Instructions

## Memory Guidelines
- Search memory at the start of any task that might benefit from past context: `memory_search`
- When the user states a preference, decision, or important fact, write it to memory: `memory_write`
- Use tier "long_term" for durable facts and decisions
- Use tier "daily" for running notes, task progress, and session context
- Record relationships between people, teams, and systems with `memory_relate`
- Before answering questions about people or organizations, check relations: `memory_search`
"""

_DEFAULT_RELATIONS_MD = """\
# Relations

Named relationships between entities (people, teams, systems, concepts).
Format: [Subject] --predicate--> [Object] (date) — note

"""


class Workspace:
    """
    Represents a single OpenMemory workspace on disk.

    Directory layout:
        <workspace_path>/
        ├── MEMORY.md           long-term curated memory
        ├── USER.md             stable user profile
        ├── AGENTS.md           agent operating instructions
        ├── RELATIONS.md        entity relation graph (human-readable mirror)
        ├── daily/
        │   └── YYYY-MM-DD.md   append-only daily logs
        └── .index/
            └── memory.db       SQLite index (vector + FTS + relations)
    """

    def __init__(self, workspace_path: Path) -> None:
        self.workspace_path = Path(workspace_path)
        self.path = self.workspace_path  # alias for backward-compat
        self._ensure_layout()

    # ------------------------------------------------------------------
    # Directory paths
    # ------------------------------------------------------------------

    @property
    def daily_dir(self) -> Path:
        return self.path / "daily"

    @property
    def index_dir(self) -> Path:
        return self.path / ".index"

    @property
    def db_path(self) -> Path:
        return self.index_dir / "memory.db"

    # ------------------------------------------------------------------
    # File paths
    # ------------------------------------------------------------------

    @property
    def memory_file(self) -> Path:
        return self.path / "MEMORY.md"

    @property
    def user_file(self) -> Path:
        return self.path / "USER.md"

    @property
    def agents_file(self) -> Path:
        return self.path / "AGENTS.md"

    @property
    def relations_file(self) -> Path:
        return self.path / "RELATIONS.md"

    def daily_file(self, day: date | None = None) -> Path:
        """Return the path for the daily log of *day* (defaults to today)."""
        d = day or date.today()
        return self.daily_dir / f"{d.isoformat()}.md"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _ensure_layout(self) -> None:
        """Create directories and seed default files if they don't exist."""
        self.path.mkdir(parents=True, exist_ok=True)
        self.daily_dir.mkdir(exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)

        self._seed(self.memory_file, _DEFAULT_MEMORY_MD)
        self._seed(self.user_file, _DEFAULT_USER_MD)
        self._seed(self.agents_file, _DEFAULT_AGENTS_MD)
        self._seed(self.relations_file, _DEFAULT_RELATIONS_MD)

    @staticmethod
    def _seed(path: Path, content: str) -> None:
        if not path.exists():
            path.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def all_memory_files(self) -> list[Path]:
        """Return all Markdown files that should be indexed."""
        files: list[Path] = []
        if self.memory_file.exists():
            files.append(self.memory_file)
        if self.relations_file.exists():
            files.append(self.relations_file)
        if self.daily_dir.exists():
            files.extend(sorted(self.daily_dir.glob("*.md")))
        return files

    def resolve_file(self, name: str) -> Path:
        """
        Resolve a user-supplied filename to an absolute path within the workspace.
        Accepts bare names like 'MEMORY.md', 'daily/2026-03-20.md', or full paths.
        """
        p = Path(name)
        if p.is_absolute():
            return p
        candidate = self.path / p
        return candidate