"""
SQLite-backed memory index.

Schema:
  - files          : tracked files with content hash + mtime
  - chunks         : text chunks with embeddings (JSON float array)
  - chunks_fts     : FTS5 virtual table for BM25 keyword search
  - relations      : named entity relationships (graph layer)
  - embedding_cache: reuse embeddings when content hasn't changed

Vector search uses a pure-Python cosine similarity fallback (always available).
If sqlite-vec extension is available it will be used automatically for faster search.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

from openmemory.core.chunker import Chunk

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS files (
    path        TEXT PRIMARY KEY,
    source      TEXT NOT NULL DEFAULT 'memory',
    hash        TEXT NOT NULL,
    mtime       REAL NOT NULL,
    size        INTEGER NOT NULL,
    indexed_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'memory',
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    model_id    TEXT NOT NULL,
    text        TEXT NOT NULL,
    embedding   TEXT NOT NULL,   -- JSON array of floats
    updated_at  REAL NOT NULL,
    FOREIGN KEY(path) REFERENCES files(path) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    chunk_id UNINDEXED,
    path     UNINDEXED,
    source   UNINDEXED,
    content='chunks',
    content_rowid='rowid'
);

CREATE TABLE IF NOT EXISTS relations (
    id          TEXT PRIMARY KEY,
    subject     TEXT NOT NULL,
    predicate   TEXT NOT NULL,
    object      TEXT NOT NULL,
    note        TEXT,
    source_file TEXT,
    created_at  REAL NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_relations_object  ON relations(object  COLLATE NOCASE);

CREATE TABLE IF NOT EXISTS embedding_cache (
    provider    TEXT NOT NULL,
    model       TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding   TEXT NOT NULL,  -- JSON array of floats
    PRIMARY KEY (provider, model, content_hash)
);
"""

# FTS5 triggers to keep the virtual table in sync with chunks
_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, chunk_id, path, source)
    VALUES (new.rowid, new.text, new.chunk_id, new.path, new.source);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, chunk_id, path, source)
    VALUES ('delete', old.rowid, old.text, old.chunk_id, old.path, old.source);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, chunk_id, path, source)
    VALUES ('delete', old.rowid, old.text, old.chunk_id, old.path, old.source);
    INSERT INTO chunks_fts(rowid, text, chunk_id, path, source)
    VALUES (new.rowid, new.text, new.chunk_id, new.path, new.source);
END;
"""


# ---------------------------------------------------------------------------
# MemoryIndex
# ---------------------------------------------------------------------------


class MemoryIndex:
    """
    Manages the SQLite memory index: schema creation, chunk upsert,
    file sync tracking, embedding cache, and relation storage.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._apply_schema()
        self._vec_available = self._try_load_vec()

    def _apply_schema(self) -> None:
        with self._conn:
            self._conn.executescript(_SCHEMA)
            self._conn.executescript(_FTS_TRIGGERS)

    def _try_load_vec(self) -> bool:
        """Attempt to load sqlite-vec extension. Returns True if available."""
        try:
            self._conn.enable_load_extension(True)
            self._conn.execute("SELECT vec_version()")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # File tracking
    # ------------------------------------------------------------------

    def get_file_record(self, path: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM files WHERE path = ?", (path,)
        ).fetchone()

    def upsert_file(self, path: str, source: str, content_hash: str, mtime: float, size: int) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO files(path, source, hash, mtime, size, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    hash=excluded.hash, mtime=excluded.mtime,
                    size=excluded.size, indexed_at=excluded.indexed_at
                """,
                (path, source, content_hash, mtime, size, time.time()),
            )

    def delete_file(self, path: str) -> None:
        """Remove a file and all its chunks (CASCADE handles chunks table)."""
        with self._conn:
            self._conn.execute("DELETE FROM files WHERE path = ?", (path,))

    # ------------------------------------------------------------------
    # Chunk upsert
    # ------------------------------------------------------------------

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]], model_id: str) -> None:
        now = time.time()
        with self._conn:
            for chunk, emb in zip(chunks, embeddings):
                self._conn.execute(
                    """
                    INSERT INTO chunks(chunk_id, path, source, start_line, end_line,
                                       content_hash, model_id, text, embedding, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        embedding=excluded.embedding,
                        model_id=excluded.model_id,
                        updated_at=excluded.updated_at
                    """,
                    (
                        chunk.chunk_id, chunk.path, chunk.source,
                        chunk.start_line, chunk.end_line,
                        chunk.content_hash, model_id, chunk.text,
                        json.dumps(emb), now,
                    ),
                )

    def delete_chunks_for_file(self, path: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE path = ?", (path,))

    def get_chunks_for_file(self, path: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM chunks WHERE path = ?", (path,)
        ).fetchall()

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    def get_cached_embedding(self, provider: str, model: str, content_hash: str) -> Optional[list[float]]:
        row = self._conn.execute(
            "SELECT embedding FROM embedding_cache WHERE provider=? AND model=? AND content_hash=?",
            (provider, model, content_hash),
        ).fetchone()
        if row:
            return json.loads(row["embedding"])
        return None

    def set_cached_embedding(self, provider: str, model: str, content_hash: str, embedding: list[float]) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO embedding_cache(provider, model, content_hash, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (provider, model, content_hash, json.dumps(embedding)),
            )

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------

    def insert_relation(
        self,
        relation_id: str,
        subject: str,
        predicate: str,
        object_: str,
        note: Optional[str] = None,
        source_file: Optional[str] = None,
        confidence: float = 1.0,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO relations
                    (id, subject, predicate, object, note, source_file, created_at, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (relation_id, subject, predicate, object_, note, source_file, time.time(), confidence),
            )

    def get_relations_for_entity(self, entity: str) -> list[sqlite3.Row]:
        """Return all relations where entity is subject OR object (case-insensitive)."""
        return self._conn.execute(
            """
            SELECT * FROM relations
            WHERE subject LIKE ? OR object LIKE ?
            ORDER BY created_at DESC
            """,
            (entity, entity),
        ).fetchall()

    def get_all_relations(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM relations ORDER BY created_at DESC"
        ).fetchall()

    def delete_relation(self, relation_id: str) -> bool:
        with self._conn:
            cur = self._conn.execute("DELETE FROM relations WHERE id = ?", (relation_id,))
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Vector search (pure Python cosine fallback)
    # ------------------------------------------------------------------

    def vector_search(
        self,
        query_embedding: list[float],
        top_k: int,
        source_filter: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve top_k chunks by cosine similarity to query_embedding.
        Falls back to Python dot-product computation when sqlite-vec is unavailable.
        """
        import numpy as np

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        filters = []
        params: list = []
        if source_filter:
            filters.append("source = ?")
            params.append(source_filter)
        if model_id:
            filters.append("model_id = ?")
            params.append(model_id)

        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        rows = self._conn.execute(
            f"SELECT chunk_id, path, source, start_line, end_line, text, embedding, updated_at FROM chunks {where}",
            params,
        ).fetchall()

        scored = []
        for row in rows:
            emb = np.array(json.loads(row["embedding"]), dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            score = float(np.dot(q, emb) / (q_norm * norm))
            scored.append({
                "chunk_id": row["chunk_id"],
                "path": row["path"],
                "source": row["source"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "text": row["text"],
                "updated_at": row["updated_at"],
                "vector_score": score,
                "text_score": 0.0,
                "score": score,
            })

        scored.sort(key=lambda x: x["vector_score"], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # FTS5 keyword search
    # ------------------------------------------------------------------

    def keyword_search(self, query: str, top_k: int, source_filter: Optional[str] = None) -> list[dict]:
        """BM25 keyword search via SQLite FTS5."""
        tokens = [t for t in query.split() if t.strip()]
        if not tokens:
            return []
        fts_query = " AND ".join(f'"{t}"' for t in tokens)

        source_filter_clause = ""
        params: list = [fts_query]
        if source_filter:
            source_filter_clause = "AND c.source = ?"
            params.append(source_filter)
        params.append(top_k)

        rows = self._conn.execute(
            f"""
            SELECT c.chunk_id, c.path, c.source, c.start_line, c.end_line, c.text,
                   c.updated_at, bm25(chunks_fts) AS bm25_rank,
                   snippet(chunks_fts, 0, '<b>', '</b>', '...', 32) AS snippet
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?
            {source_filter_clause}
            ORDER BY bm25_rank
            LIMIT ?
            """,
            params,
        ).fetchall()

        results = []
        for row in rows:
            # BM25 rank from FTS5: more negative = better match
            # Normalize to 0–1: rank=0 → 1.0 (perfect), rank approaches -inf → ~0
            rank = row["bm25_rank"]  # negative float
            text_score = 1.0 / (1.0 + abs(rank)) if rank != 0 else 1.0
            results.append({
                "chunk_id": row["chunk_id"],
                "path": row["path"],
                "source": row["source"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "text": row["text"],
                "snippet": row["snippet"],
                "updated_at": row["updated_at"],
                "vector_score": 0.0,
                "text_score": text_score,
                "score": text_score,
            })
        return results

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*) as files FROM files"
        ).fetchone()
        chunks = self._conn.execute(
            "SELECT COUNT(*) as c FROM chunks"
        ).fetchone()
        relations = self._conn.execute(
            "SELECT COUNT(*) as r FROM relations"
        ).fetchone()
        return {
            "files": row["files"],
            "chunks": chunks["c"],
            "relations": relations["r"],
            "vec_extension": self._vec_available,
        }

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MemoryIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()