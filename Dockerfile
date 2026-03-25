# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build argument - set EXTRAS=local to include sentence-transformers
# (adds ~1 GB; only needed for local offline embeddings)
#
# Examples:
#   docker build .                          # BM25-only + OpenAI-compatible API
#   docker build --build-arg EXTRAS=local . # + sentence-transformers
# ---------------------------------------------------------------------------
ARG EXTRAS=""
ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------
# Stage 1: builder - installs gcc + all Python deps into a venv
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ARG EXTRAS

# Install build deps in a single layer; no cleanup needed (stage is discarded)
RUN apt-get update && apt-get install -y --no-install-recommends gcc

WORKDIR /app

# Create an isolated venv so we can copy it cleanly to the runtime stage
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY groundmemory/ ./groundmemory/

RUN if [ -n "$EXTRAS" ]; then \
        pip install --no-cache-dir ".[$EXTRAS]"; \
    else \
        pip install --no-cache-dir .; \
    fi

# ---------------------------------------------------------------------------
# Stage 2: runtime - clean image, no compiler, no build cache
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

# Don't run as root
RUN useradd --create-home appuser

# Pull in only the installed venv from the builder stage.
# The package is fully installed inside /venv - no source tree copy needed.
COPY --from=builder /venv /venv

WORKDIR /app

ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GROUNDMEMORY_ROOT_DIR=/data \
    GROUNDMEMORY_WORKSPACE=default \
    GROUNDMEMORY_EMBEDDING__PROVIDER=none

EXPOSE 4242

VOLUME ["/data"]

USER appuser

CMD ["groundmemory-mcp"]