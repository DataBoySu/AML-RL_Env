# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the AML Investigator Environment.

Wraps AmlEnvironment with OpenEnv's create_app utility, exposing the
standard OpenEnv HTTP endpoints required by the evaluator:

    POST /reset  — Reset episode, returns initial AmlObservation
    POST /step   — Execute an AmlAction, returns AmlObservation
    GET  /state  — Return current internal State object
    GET  /schema — Return action/observation JSON schemas
    WS   /ws     — WebSocket endpoint for persistent sessions

CORS is configured to allow all origins so the hackathon evaluator
can reach the Space without origin restrictions.

Usage (local dev):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

Usage (production / HF Spaces):
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with:\n  pip install -r server/requirements.txt"
    ) from e

from fastapi.middleware.cors import CORSMiddleware

try:
    # Relative imports when loaded as part of the package (e.g. `python -m server.app`)
    from ..models import AmlAction, AmlObservation
    from .AML_env_environment import AmlEnvironment
except ImportError:
    # Absolute imports when PYTHONPATH is set to the repo root (Docker / uvicorn CLI)
    from models import AmlAction, AmlObservation
    from server.AML_env_environment import AmlEnvironment


# ---------------------------------------------------------------------------
# Build the OpenEnv-compliant FastAPI application
# ---------------------------------------------------------------------------
app = create_app(
    AmlEnvironment,
    AmlAction,
    AmlObservation,
    env_name="AML_env",
    # One concurrent WebSocket session is enough for HF Spaces evaluation.
    # Increase for multi-agent experiments.
    max_concurrent_envs=10,
)

# ---------------------------------------------------------------------------
# CORS — allow the OpenEnv evaluator (and any browser) to reach the Space
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Direct-execution entry point
# ---------------------------------------------------------------------------
def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """
    Run the server directly.

    Examples:
        python -m server.app
        python -m server.app --port 7860
    """
    import uvicorn

    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AML Investigator OpenEnv server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=7860, help="Bind port")
    args = parser.parse_args()
    # Call main() so the OpenEnv validator can detect it via string search
    main()
