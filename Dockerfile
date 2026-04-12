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

FROM python:3.11-slim

# --- System dependencies -------------------------------------------------
# curl  → healthcheck
# git   → uv may resolve VCS dependencies (openenv from git)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# --- Install uv ----------------------------------------------------------
# uv is the canonical package manager for this project (see uv.lock).
# We download the pre-built binary so Docker layer caching is fast.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# --- Working directory ---------------------------------------------------
WORKDIR /app

# --- Copy full project context -------------------------------------------
# Copy everything so uv sync can resolve the full project graph.
# Unwanted paths are excluded via .dockerignore.
COPY . /app/

# --- Install dependencies via uv -----------------------------------------
# Use --frozen to honour the checked-in uv.lock for reproducibility.
# Falls back to a live resolve if uv.lock is absent (shouldn't happen).
RUN if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# --- Runtime environment -------------------------------------------------
# Add the uv-managed venv to PATH so uvicorn / python resolve correctly.
ENV PATH="/app/.venv/bin:$PATH"

# PYTHONPATH → repo root so that both of these import patterns work:
#   from models import AmlAction           (absolute, no package prefix)
#   from server.AML_env_environment import AmlEnvironment
ENV PYTHONPATH="/app"

# Hugging Face Spaces mandates port 7860.
EXPOSE 7860

# --- Health check --------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# --- Start server --------------------------------------------------------
# Module path: server/app.py → server.app:app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
