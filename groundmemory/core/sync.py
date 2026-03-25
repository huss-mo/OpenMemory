"""
File synchronization: keeps the SQLite index in sync with the workspace Markdown files.

Uses SHA-256 content hashing (not timestamps) to detect changes - reliable across
file copies, moves, and system clock changes.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional

from groundmemory.config import ChunkingConfig
from groundmemory.core.chunker import chunk_file
from groundmemory.core.embeddings import EmbeddingProvider
from groundmemory.core.relations import sync_relations_from_file
from groundmemory.core.index import MemoryIndex
from groundmemory.core.workspace import Workspace


def _file_hash(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()


def sync_workspace(
    workspace: Workspace,
    index: MemoryIndex,
    provider: EmbeddingProvider,
    chunking: ChunkingConfig,
    force: bool = False,
) -> dict:
    """
    Walk all memory files in the workspace and update the index for any
    files that have been added, modified, or deleted since last sync.

    Args:
        workspace: Workspace instance.
        index:     MemoryIndex instance.
        provider:  EmbeddingProvider for generating embeddings.
        chunking:  ChunkingConfig.
        force:     If True, re-index all files regardless of hash.

    Returns:
        A summary dict: {added, updated, deleted, skipped, errors}
    """
    summary = {"added": 0, "updated": 0, "deleted": 0, "skipped": 0, "errors": []}

    current_files = {str(p): p for p in workspace.all_memory_files()}

    # --- Detect deleted files ---
    indexed_paths = {
        row["path"]
        for row in index._conn.execute("SELECT path FROM files").fetchall()
    }
    for path_str in indexed_paths:
        if path_str not in current_files:
            index.delete_file(path_str)
            summary["deleted"] += 1

    # --- Add / update changed files ---
    for path_str, path in current_files.items():
        try:
            content_hash = _file_hash(path)
            stat = path.stat()
            existing = index.get_file_record(path_str)

            if not force and existing and existing["hash"] == content_hash:
                summary["skipped"] += 1
                continue

            # Determine source label
            name = path.name.upper()
            if name == "MEMORY.MD":
                source = "long_term"
            elif name == "RELATIONS.MD":
                source = "relations"
            elif name == "USER.MD":
                source = "user"
            elif name == "AGENTS.MD":
                source = "agents"
            elif "daily" in path.parts:
                source = "daily"
            else:
                source = "memory"

            # Chunk the file
            chunks = chunk_file(path, chunking)
            if not chunks:
                summary["skipped"] += 1
                continue

            # Embed with cache
            texts = [c.text for c in chunks]
            embeddings = _embed_with_cache(texts, provider, index)

            # Delete old chunks for this file, then insert new ones
            # upsert_file MUST come before upsert_chunks (FK constraint)
            index.delete_chunks_for_file(path_str)
            index.upsert_file(
                path=path_str,
                source=source,
                content_hash=content_hash,
                mtime=stat.st_mtime,
                size=stat.st_size,
            )
            index.upsert_chunks(chunks, embeddings, provider.model_id)

            # Keep relations table in sync when RELATIONS.md changes
            if name == "RELATIONS.MD":
                sync_relations_from_file(path, index)

            if existing:
                summary["updated"] += 1
            else:
                summary["added"] += 1

        except Exception as e:
            summary["errors"].append({"file": path_str, "error": str(e)})

    return summary


def sync_file(
    path: Path,
    index: MemoryIndex,
    provider: EmbeddingProvider,
    chunking: ChunkingConfig,
) -> dict:
    """
    Force-sync a single file into the index.
    Used after memory_write to immediately make new content searchable.
    """
    if not path.exists():
        return {"error": f"File not found: {path}"}

    try:
        content_hash = _file_hash(path)
        stat = path.stat()

        name = path.name.upper()
        if name == "MEMORY.MD":
            source = "long_term"
        elif name == "RELATIONS.MD":
            source = "relations"
        elif name == "USER.MD":
            source = "user"
        elif name == "AGENTS.MD":
            source = "agents"
        elif "daily" in path.parts:
            source = "daily"
        else:
            source = "memory"

        chunks = chunk_file(path, chunking)
        if not chunks:
            return {"status": "empty", "file": str(path)}

        embeddings = _embed_with_cache([c.text for c in chunks], provider, index)
        index.delete_chunks_for_file(str(path))
        # upsert_file MUST come before upsert_chunks (FK constraint)
        index.upsert_file(
            path=str(path),
            source=source,
            content_hash=content_hash,
            mtime=stat.st_mtime,
            size=stat.st_size,
        )
        index.upsert_chunks(chunks, embeddings, provider.model_id)

        # Keep relations table in sync when RELATIONS.md changes
        if name == "RELATIONS.MD":
            sync_relations_from_file(path, index)

        return {"status": "synced", "file": str(path), "chunks": len(chunks)}
    except Exception as e:
        return {"status": "error", "file": str(path), "error": str(e)}


def _embed_with_cache(
    texts: list[str],
    provider: EmbeddingProvider,
    index: MemoryIndex,
) -> list[list[float]]:
    """
    Embed *texts* using the cache in the index.
    Only calls the provider for texts whose hash isn't cached.
    """
    import hashlib

    provider_name = provider.model_id.split("/")[0]
    model_name = provider.model_id

    results: list[Optional[list[float]]] = [None] * len(texts)
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for i, text in enumerate(texts):
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        cached = index.get_cached_embedding(provider_name, model_name, content_hash)
        if cached is not None:
            results[i] = cached
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        new_embeddings = provider.embed(uncached_texts)
        for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            index.set_cached_embedding(provider_name, model_name, content_hash, emb)
            results[idx] = emb

    return [r for r in results if r is not None]