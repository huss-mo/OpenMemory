"""
Markdown-aware text chunker.

Splits content into overlapping chunks that respect Markdown heading boundaries.
Each chunk records its source line range for precise memory_get retrieval.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openmemory.config import ChunkingConfig


@dataclass
class Chunk:
    """A single text chunk extracted from a Markdown file."""

    chunk_id: str          # SHA-256 of (path + start_line + text)
    path: str              # absolute path of the source file
    source: str            # "memory", "daily", "relations", etc.
    start_line: int        # 0-indexed, inclusive
    end_line: int          # 0-indexed, exclusive
    text: str              # chunk content
    content_hash: str      # SHA-256 of text (for dedup / change detection)

    @classmethod
    def make(cls, path: str, source: str, start: int, end: int, text: str) -> "Chunk":
        raw = f"{path}:{start}:{text}"
        chunk_id = hashlib.sha256(raw.encode()).hexdigest()
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        return cls(
            chunk_id=chunk_id,
            path=path,
            source=source,
            start_line=start,
            end_line=end,
            text=text,
            content_hash=content_hash,
        )


def _classify_source(path: str) -> str:
    """Derive a source label from the file path."""
    p = Path(path)
    name = p.name.upper()
    if name == "MEMORY.MD":
        return "long_term"
    if name == "RELATIONS.MD":
        return "relations"
    if name == "USER.MD":
        return "user"
    if name == "AGENTS.MD":
        return "agents"
    # Anything under daily/ directory
    if "daily" in p.parts:
        return "daily"
    return "memory"


def chunk_markdown(
    text: str,
    path: str,
    config: ChunkingConfig,
    source: Optional[str] = None,
) -> list[Chunk]:
    """
    Split *text* (from *path*) into overlapping chunks.

    Strategy:
    1. Split on lines, tracking line numbers.
    2. Heading lines (# / ## / ###) act as hard chunk boundaries —
       a new heading always starts a fresh chunk.
    3. When accumulated characters exceed max_chars, flush the current chunk
       and carry *overlap_chars* worth of recent lines into the next one.

    Returns a list of Chunk objects with 0-indexed line ranges.
    """
    if not text.strip():
        return []

    src = source or _classify_source(path)
    max_chars = max(32, config.tokens * 4)       # approx: 1 token ≈ 4 chars
    overlap_chars = max(0, config.overlap * 4)

    lines = text.splitlines(keepends=True)
    chunks: list[Chunk] = []

    # Current accumulator state
    buf_lines: list[str] = []
    buf_start: int = 0
    buf_chars: int = 0

    def flush(buf: list[str], start: int) -> None:
        chunk_text = "".join(buf).strip()
        if chunk_text:
            end = start + len(buf)
            chunks.append(Chunk.make(path, src, start, end, chunk_text))

    for i, line in enumerate(lines):
        is_heading = line.startswith("#")

        # Hard boundary: flush before any heading (unless buffer is empty)
        if is_heading and buf_chars > 0:
            flush(buf_lines, buf_start)
            # Carry overlap: take trailing lines that fit within overlap_chars
            overlap_lines: list[str] = []
            carried = 0
            for ol in reversed(buf_lines):
                if carried + len(ol) > overlap_chars:
                    break
                overlap_lines.insert(0, ol)
                carried += len(ol)
            buf_lines = overlap_lines
            buf_start = i - len(buf_lines)
            buf_chars = carried

        # Soft boundary: flush when buffer exceeds max_chars
        if buf_chars > 0 and buf_chars + len(line) > max_chars and not is_heading:
            flush(buf_lines, buf_start)
            overlap_lines = []
            carried = 0
            for ol in reversed(buf_lines):
                if carried + len(ol) > overlap_chars:
                    break
                overlap_lines.insert(0, ol)
                carried += len(ol)
            buf_lines = overlap_lines
            buf_start = i - len(buf_lines)
            buf_chars = carried

        # Very long single lines: hard-split at max_chars
        if len(line) > max_chars:
            if buf_chars > 0:
                flush(buf_lines, buf_start)
                buf_lines = []
                buf_start = i
                buf_chars = 0
            # Split the line itself
            for offset in range(0, len(line), max_chars - overlap_chars):
                segment = line[offset : offset + max_chars]
                chunks.append(Chunk.make(path, src, i, i + 1, segment.strip()))
            continue

        buf_lines.append(line)
        buf_chars += len(line)

    # Flush the final buffer
    if buf_lines:
        flush(buf_lines, buf_start)

    return chunks


def chunk_file(file_path: Path, config: ChunkingConfig) -> list[Chunk]:
    """Convenience wrapper: read a file and chunk it."""
    if not file_path.exists():
        return []
    text = file_path.read_text(encoding="utf-8")
    return chunk_markdown(text, str(file_path), config)