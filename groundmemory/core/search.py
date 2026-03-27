"""
Hybrid search: BM25 keyword + vector cosine similarity with graph-aware expansion.

Pipeline:
  1. Embed the query
  2. Run vector search (cosine) → top_k * candidate_multiplier candidates
  3. Run FTS5 keyword search → top_k * candidate_multiplier candidates
  4. Merge & re-score: score = vector_weight * vec_score + (1-vector_weight) * text_score
  5. Cross-encoder reranking (optional, if rerank_model is set)
  6. Apply temporal decay (if configured) — post-rerank so recency nudges relevance scores
  7. MMR diversification (optional, if mmr_lambda > 0) — greedily selects top_k results
     that balance relevance against similarity to already-selected results
  8. Graph expansion: extract entity mentions from top results, pull in related relations
  9. Return top_k final results
"""

from __future__ import annotations

import math
import re
import time
from typing import Optional

import numpy as np

from groundmemory.config import SearchConfig
from groundmemory.core.embeddings import EmbeddingProvider
from groundmemory.core.index import MemoryIndex


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


class SearchResult:
    """A single search result with full provenance."""

    __slots__ = (
        "chunk_id", "path", "source", "start_line", "end_line",
        "text", "snippet", "score", "vector_score", "text_score",
        "updated_at", "relation_context",
    )

    def __init__(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, "snippet"):
            self.snippet = None
        if not hasattr(self, "relation_context"):
            self.relation_context = []

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "path": self.path,
            "source": self.source,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
            "snippet": self.snippet,
            "score": round(self.score, 4),
            "vector_score": round(self.vector_score, 4),
            "text_score": round(self.text_score, 4),
            "relation_context": self.relation_context,
        }


# ---------------------------------------------------------------------------
# Helper: extract likely entity names from text
# ---------------------------------------------------------------------------

_ENTITY_PATTERN = re.compile(r"\[([^\]]+)\]|\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")


def _extract_entities(text: str) -> list[str]:
    """
    Heuristic entity extraction:
    - Markdown link-style entities: [Alice], [Auth Team]
    - Capitalised proper nouns: Alice, Auth Service
    """
    found = set()
    for m in _ENTITY_PATTERN.finditer(text):
        entity = m.group(1) or m.group(2)
        if entity and len(entity) > 1:
            found.add(entity.strip())
    return list(found)


# ---------------------------------------------------------------------------
# Hybrid merge
# ---------------------------------------------------------------------------


def _merge_results(
    vector_results: list[dict],
    keyword_results: list[dict],
    vector_weight: float,
) -> list[dict]:
    """Merge vector and keyword result lists into a single scored list."""
    text_weight = 1.0 - vector_weight
    by_id: dict[str, dict] = {}

    for r in vector_results:
        by_id[r["chunk_id"]] = {**r, "text_score": 0.0, "snippet": None}

    for r in keyword_results:
        cid = r["chunk_id"]
        if cid in by_id:
            by_id[cid]["text_score"] = r["text_score"]
            by_id[cid]["snippet"] = r.get("snippet")
        else:
            by_id[cid] = {**r, "vector_score": 0.0}

    merged = []
    for item in by_id.values():
        item["score"] = (
            vector_weight * item["vector_score"]
            + text_weight * item["text_score"]
        )
        merged.append(item)

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged


# ---------------------------------------------------------------------------
# Temporal decay
# ---------------------------------------------------------------------------


def _apply_temporal_decay(results: list[dict], decay_rate: float) -> list[dict]:
    """
    Multiply each score by exp(-decay_rate * days_since_update).
    decay_rate=0 disables the effect.
    """
    if decay_rate == 0.0:
        return results
    now = time.time()
    for r in results:
        age_days = (now - r.get("updated_at", now)) / 86400.0
        r["score"] = r["score"] * math.exp(-decay_rate * age_days)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# MMR diversification
# ---------------------------------------------------------------------------


def _apply_mmr(
    results: list[dict],
    top_k: int,
    mmr_lambda: float,
    index: MemoryIndex,
) -> list[dict]:
    """
    Maximal Marginal Relevance (MMR) diversification.

    Greedily selects ``top_k`` results from ``results`` by iteratively picking
    the candidate that maximises:

        mmr_lambda * relevance_score  -  (1 - mmr_lambda) * max_cosine_sim_to_selected

    where ``relevance_score`` is the hybrid (post-decay) score and
    ``max_cosine_sim_to_selected`` is the maximum cosine similarity to any
    result already selected (computed from stored chunk embeddings).

    Args:
        results:    Candidate list, already sorted by descending score.
        top_k:      Number of results to select.
        mmr_lambda: Trade-off weight.  0.0 = pure diversity; 1.0 = pure relevance.
                    Caller guards against mmr_lambda == 0.0 (disabled).
        index:      MemoryIndex used to retrieve stored chunk embeddings.

    Returns:
        A new list of at most ``top_k`` result dicts in MMR-selected order.
    """
    if not results:
        return results

    # Fetch embeddings for all candidates in a single bulk DB query.
    chunk_ids = [r["chunk_id"] for r in results]
    emb_map = index.get_embeddings_by_ids(chunk_ids)

    # Build a parallel list of unit-normalised numpy vectors.
    # Candidates without a stored embedding are treated as having zero
    # similarity to everything (they will only be selected for relevance).
    vecs: list[Optional[np.ndarray]] = []
    for r in results:
        raw = emb_map.get(r["chunk_id"])
        if raw is not None:
            v = np.array(raw, dtype=np.float32)
            norm = np.linalg.norm(v)
            vecs.append(v / norm if norm > 0 else None)
        else:
            vecs.append(None)

    selected_indices: list[int] = []
    remaining_indices = list(range(len(results)))

    while remaining_indices and len(selected_indices) < top_k:
        best_idx: Optional[int] = None
        best_mmr_score = float("-inf")

        for i in remaining_indices:
            relevance = results[i]["score"]

            # Max cosine similarity to already-selected items.
            if not selected_indices:
                max_sim = 0.0
            else:
                vi = vecs[i]
                sims: list[float] = []
                for j in selected_indices:
                    vj = vecs[j]
                    if vi is not None and vj is not None:
                        sims.append(float(np.dot(vi, vj)))
                    else:
                        sims.append(0.0)
                max_sim = max(sims) if sims else 0.0

            mmr_score = mmr_lambda * relevance - (1.0 - mmr_lambda) * max_sim
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = i

        if best_idx is None:
            break
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return [results[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# Graph expansion
# ---------------------------------------------------------------------------


def _expand_with_relations(
    results: list[dict],
    index: MemoryIndex,
    top_k: int,
) -> list[dict]:
    """
    For the top results, extract entity mentions and attach related relations
    as `relation_context` on each result. Does not add new results - enriches
    existing ones with relational knowledge.
    """
    # Collect entities from top-k results
    all_entities: set[str] = set()
    for r in results[:top_k]:
        all_entities.update(_extract_entities(r.get("text", "")))

    if not all_entities:
        return results

    # Build a relation lookup
    entity_relations: dict[str, list[str]] = {}
    for entity in all_entities:
        rows = index.get_relations_for_entity(entity)
        if rows:
            entity_relations[entity] = [
                f"[{row['subject']}] --{row['predicate']}--> [{row['object']}]"
                + (f" - {row['note']}" if row["note"] else "")
                for row in rows
            ]

    if not entity_relations:
        return results

    # Attach to relevant results
    for r in results[:top_k]:
        entities_in_chunk = _extract_entities(r.get("text", ""))
        context: list[str] = []
        seen: set[str] = set()
        for e in entities_in_chunk:
            for rel_str in entity_relations.get(e, []):
                if rel_str not in seen:
                    context.append(rel_str)
                    seen.add(rel_str)
        r["relation_context"] = context

    return results


# ---------------------------------------------------------------------------
# Main search function
# ---------------------------------------------------------------------------


def hybrid_search(
    query: str,
    index: MemoryIndex,
    provider: EmbeddingProvider,
    config: SearchConfig,
    source_filter: Optional[str] = None,
    top_k: Optional[int] = None,
) -> list[SearchResult]:
    """
    Full hybrid search pipeline:
      embed → vector search + keyword search → merge → rerank → decay → MMR → graph expand → top_k

    Args:
        query:         Natural language search query.
        index:         MemoryIndex instance.
        provider:      Embedding provider.
        config:        SearchConfig with weights, top_k, decay settings.
        source_filter: Optional source label to restrict search scope.
        top_k:         Override config.top_k if provided.

    Returns:
        List of SearchResult objects, sorted by descending score.
    """
    from groundmemory.core.reranker import rerank  # local import to keep reranker optional

    k = top_k or config.top_k
    candidates = k * config.candidate_multiplier

    # Step 1: Embed query
    query_embeddings = provider.embed([query])
    if not query_embeddings:
        return []
    query_vec = query_embeddings[0]

    # Step 2: Vector search
    vec_results = index.vector_search(
        query_embedding=query_vec,
        top_k=candidates,
        source_filter=source_filter,
        model_id=provider.model_id,
    )

    # Step 3: Keyword search
    kw_results = index.keyword_search(
        query=query,
        top_k=candidates,
        source_filter=source_filter,
    )

    # Step 4: Merge
    merged = _merge_results(vec_results, kw_results, config.vector_weight)

    # Step 5: Cross-encoder reranking (optional)
    if config.rerank_model:
        merged = rerank(query, merged, config.rerank_model)

    # Step 6: Temporal decay
    merged = _apply_temporal_decay(merged, config.temporal_decay_rate)

    # Step 7: MMR diversification (optional)
    if config.mmr_lambda > 0.0:
        merged = _apply_mmr(merged, k, config.mmr_lambda, index)

    # Step 8: Graph expansion (enrich top results with relation context)
    merged = _expand_with_relations(merged, index, k)

    # Step 9: Return top_k as SearchResult objects
    final = merged[:k]
    return [
        SearchResult(
            chunk_id=r["chunk_id"],
            path=r["path"],
            source=r["source"],
            start_line=r["start_line"],
            end_line=r["end_line"],
            text=r["text"],
            snippet=r.get("snippet"),
            score=r["score"],
            vector_score=r["vector_score"],
            text_score=r["text_score"],
            updated_at=r.get("updated_at", 0.0),
            relation_context=r.get("relation_context", []),
        )
        for r in final
    ]