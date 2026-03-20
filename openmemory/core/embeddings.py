"""
Embedding provider abstraction.

Two providers:
  - SentenceTransformerProvider: local, free, uses sentence-transformers library
  - OpenAICompatibleProvider: any OpenAI-compatible HTTP API (OpenAI, LiteLLM, Ollama, etc.)

Auto-selection: if OPENMEMORY_EMBEDDING_BASE_URL or OPENAI_API_KEY is set → openai;
               otherwise → local sentence-transformers.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from openmemory.config import EmbeddingConfig


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning one float vector per text."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable identifier used to detect model changes in the DB."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...


# ---------------------------------------------------------------------------
# Local provider (sentence-transformers)
# ---------------------------------------------------------------------------


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Uses the sentence-transformers library for local, offline embedding.
    Default model: all-MiniLM-L6-v2 (fast, 384-dim, ~90 MB).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: uv add --optional local sentence-transformers"
            ) from e

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        # Determine dimensions from a test embedding
        test = self._model.encode(["test"], convert_to_numpy=True)
        self._dims = int(test.shape[1])

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vecs.tolist()

    @property
    def model_id(self) -> str:
        return f"sentence-transformers/{self._model_name}"

    @property
    def dimensions(self) -> int:
        return self._dims


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP provider
# ---------------------------------------------------------------------------


class OpenAICompatibleProvider(EmbeddingProvider):
    """
    Calls any OpenAI-compatible /embeddings endpoint.
    Works with: OpenAI directly, LiteLLM proxy, Ollama, LocalAI, Mistral, etc.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        try:
            import httpx
        except ImportError as e:
            raise ImportError("httpx is required: uv add httpx") from e

        self._model = model
        self._base_url = (base_url or os.environ.get("OPENMEMORY_EMBEDDING_BASE_URL")
                          or "https://api.openai.com/v1").rstrip("/")
        self._api_key = (api_key or os.environ.get("OPENMEMORY_EMBEDDING_API_KEY")
                         or os.environ.get("OPENAI_API_KEY") or "")
        self._batch_size = batch_size
        self._dims: Optional[int] = None  # lazily determined on first call
        import httpx as _httpx
        self._client = _httpx.Client(timeout=60)

    def _call_api(self, batch: list[str]) -> list[list[float]]:
        import httpx

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        payload = {"model": self._model, "input": batch}
        try:
            resp = self._client.post(
                f"{self._base_url}/embeddings",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Embedding API error {e.response.status_code}: {e.response.text}"
            ) from e

        data = resp.json()
        # Sort by index to maintain order (some providers reorder)
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            results.extend(self._call_api(batch))
        if self._dims is None and results:
            self._dims = len(results[0])
        return results

    @property
    def model_id(self) -> str:
        return f"openai-compatible/{self._model}"

    @property
    def dimensions(self) -> int:
        if self._dims is None:
            # Trigger a test embed to discover dimensions
            self.embed(["test"])
        return self._dims or 1536  # fallback for text-embedding-3-small


# ---------------------------------------------------------------------------
# Null provider (BM25-only mode, no embeddings)
# ---------------------------------------------------------------------------


class NullEmbeddingProvider(EmbeddingProvider):
    """
    A no-op embedding provider for BM25-keyword-only deployments.
    Vector search is disabled; only FTS5 keyword search is used.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:  # noqa: ARG002
        return [[] for _ in texts]

    @property
    def model_id(self) -> str:
        return "null"

    @property
    def dimensions(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Instantiate the correct provider based on configuration."""
    if config.provider == "none":
        return NullEmbeddingProvider()
    if config.provider == "openai":
        return OpenAICompatibleProvider(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            batch_size=config.batch_size,
        )
    # Default: local sentence-transformers
    return SentenceTransformerProvider(model_name=config.local_model)


# ---------------------------------------------------------------------------
# Cosine similarity (fallback for when sqlite-vec is unavailable)
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)