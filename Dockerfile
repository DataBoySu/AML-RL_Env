# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# ============================================================
# AML Investigator — OpenEnv Environment
# Hugging Face Spaces compliant Docker image
# ============================================================
#
# Build (from repo root):
#   docker build -t aml-env .
#
# Run locally:
#   docker run -p 7860:7860 aml-env
# ============================================================

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# git is needed for uv to resolve any VCS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy full build context (unwanted files pruned by .dockerignore)
COPY . /app/env
WORKDIR /app/env

# Ensure uv is available (the openenv-base image usually has it; install as fallback)
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install deps only (no project install yet) — uses --frozen so uv.lock is honoured
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

# Install the project itself into the venv
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

# curl is required for the HEALTHCHECK; install it in the RUNTIME stage
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy venv and source from builder
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

# Create unprivileged user (good practice for HF Spaces)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# The venv bin directory must be first on PATH
ENV PATH="/app/.venv/bin:$PATH"

# PYTHONPATH → /app/env (repo root inside container)
# This makes both import styles work:
#   from models import AmlAction             (bare)
#   from server.AML_env_environment import … (prefixed)
ENV PYTHONPATH="/app/env"

# Hugging Face Spaces mandates port 7860
EXPOSE 7860

# Health check — verifiable with `docker inspect`
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

WORKDIR /app/env
USER appuser

# Start the OpenEnv FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]