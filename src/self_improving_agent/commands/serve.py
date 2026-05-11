"""``sia serve`` — start the inference HTTP server.

Implementation lives in ``self_improving_agent.serving`` (T1.1). This module
is the CLI hook; it imports lazily so ``sia --help`` doesn't pay the FastAPI
import cost.
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--registry",
        default="models/registry.json",
        help="Registry path (default: models/registry.json).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="uvicorn --reload for development.",
    )


def run(args: argparse.Namespace) -> int:
    try:
        import uvicorn  # type: ignore
    except ImportError:
        print(
            "uvicorn not installed. Install the serving extras:\n"
            "    pip install 'self-improving-agent[serving]'\n"
            "or add fastapi+uvicorn manually."
        )
        return 1

    # Lazy import keeps `sia --help` cheap and `sia status` independent.
    from self_improving_agent.serving import server  # noqa: F401

    # Pass registry path via env so the server module is import-clean.
    import os

    os.environ.setdefault("SIA_REGISTRY_PATH", args.registry)

    uvicorn.run(
        "self_improving_agent.serving.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0
