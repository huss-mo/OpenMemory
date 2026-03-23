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

### Choosing the right tier for `memory_write`

| Tier | File | Use for |
|------|------|---------|
| `long_term` | MEMORY.md | Curated facts, decisions, project knowledge — anything that should persist indefinitely |
| `daily` | daily/YYYY-MM-DD.md | Running session notes, task progress, short-lived context |
| `user` | USER.md | Stable facts *about the user*: name, role, organisation, location, preferences |
| `agent` | AGENTS.md | Behavioural rules for future sessions (e.g. "always search before answering") |

**Rules of thumb:**
- When the user reveals something about themselves → `memory_write(tier="user")`
- When the user states a preference or decision → `memory_write(tier="long_term")` or `memory_write(tier="user")` if it's personal
- When you need to remember a session task → `memory_write(tier="daily")`
- When a new behavioural rule is established → `memory_write(tier="agent")`

### Search before answering
- At the start of any task that might benefit from past context, call `memory_search`
- Before answering questions about people or organisations, check relations too

### Recording relationships
Use `memory_relate` for directional facts between entities. Always use **snake_case** predicates.

```
memory_relate(subject="Alice", predicate="works_at", object="Acme Corp")
memory_relate(subject="Auth Service", predicate="owned_by", object="Platform Team")
memory_relate(subject="Bob", predicate="manages", object="Alice")
```

When to relate vs. write:
- Structural fact about people, teams, or systems → `memory_relate`
- Free-form preference, decision, or note → `memory_write`

#### Superseding outdated relations

When a relation changes (job change, location change, team reassignment), use `supersedes=True` to replace the old value instead of accumulating duplicates:

```
# User changed jobs — old works_at should be removed
memory_relate(subject="Hussein", predicate="works_at", object="One Industry", supersedes=True, note="Previously at iHorizons")

# User moved cities
memory_relate(subject="Alice", predicate="lives_in", object="Berlin", supersedes=True)
```

**Rules:**
- Use `supersedes=True` only when the old value is no longer valid — it deletes ALL prior `(subject, predicate)` triples.
- Do NOT use `supersedes=True` when multiple objects are valid at the same time (e.g. `knows`, `attended`, `member_of`).
- Before writing any relation, call `memory_search` to check if a conflicting one already exists.

### Editing RELATIONS.md directly

You may read and edit RELATIONS.md using `memory_get`, `memory_replace_text`, `memory_replace_lines`, and `memory_delete`. **Every non-blank, non-header line must follow this exact format:**

```
- [Subject] --predicate--> [Object] (YYYY-MM-DD) — "optional note"
```

Examples of valid lines:
```
- [Alice] --leads--> [Auth Team] (2026-03-20)
- [Alice] --leads--> [Auth Team] (2026-03-20) — "Assigned during sprint planning"
- [Auth Service] --owned_by--> [Platform Team] (2026-03-20)
```

**Rules:**
- Subject and Object are wrapped in square brackets: `[Name]`
- Predicate uses **snake_case** and is wrapped with `--` and `-->`: `--predicate-->`
- Date `(YYYY-MM-DD)` is required
- Note (after `—`) is optional
- Lines that do not match this format will be rejected when using replace tools
"""

_DEFAULT_RELATIONS_MD = """\
# Relations

Named relationships between entities (people, teams, systems, concepts).
Format: [Subject] --predicate--> [Object] (date) — note

"""


class Workspace:
    """
    Represents a single groundmemory workspace on disk.

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
        if self.user_file.exists():
            files.append(self.user_file)
        if self.agents_file.exists():
            files.append(self.agents_file)
        if self.relations_file.exists():
            files.append(self.relations_file)
        if self.daily_dir.exists():
            files.extend(sorted(self.daily_dir.glob("*.md")))
        return files

    def resolve_file(self, name: str) -> Path:
        """
        Resolve a user-supplied filename to an absolute path within the workspace.
        Accepts bare names like 'MEMORY.md' or 'daily/2026-03-20.md'.

        Raises ValueError if:
        - the supplied path is absolute (no legitimate tool call needs this), or
        - the resolved path escapes the workspace directory (path traversal).
        """
        p = Path(name)
        if p.is_absolute():
            raise ValueError(
                f"Access denied: absolute paths are not allowed (got '{name}')."
            )
        candidate = (self.path / p).resolve()
        workspace_root = self.path.resolve()
        if not candidate.is_relative_to(workspace_root):
            raise ValueError(
                f"Access denied: '{name}' resolves outside the workspace."
            )
        return candidate
