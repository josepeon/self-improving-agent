"""``sia dashboard`` — launch the Streamlit eval dashboard.

Thin wrapper that resolves the dashboard module path and execs streamlit.
Streamlit is in the ``[dashboard]`` extra — not a base dep — so we lazy-import
and print a helpful error if it isn't installed.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for the dashboard (default: 8501).",
    )
    parser.add_argument(
        "--registry",
        default="models/registry.json",
        help="Registry path (default: models/registry.json).",
    )
    parser.add_argument(
        "--reports-dir",
        default="evaluation/reports",
        help="Reports directory (default: evaluation/reports).",
    )


def run(args: argparse.Namespace) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "streamlit is not installed. Install the dashboard extra:\n"
            "    pip install -e '.[dashboard]'"
        )
        return 1

    # Use module path so users don't need to know where the file lives
    from self_improving_agent.dashboard import app as dashboard_app

    app_path = Path(dashboard_app.__file__)
    env = os.environ.copy()
    env["SIA_REGISTRY_PATH"] = args.registry
    env["SIA_REPORTS_DIR"] = args.reports_dir

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path), "--server.port", str(args.port),
        "--server.headless", "true",
    ]
    return subprocess.call(cmd, env=env)
