"""Shared utilities for scripts/ — extracted so each new script doesn't
copy-paste argparse setup, env loading, logging, and progress bars.

Import with:

    from _lib import (
        init_logging, load_env_or_die, make_progress, resolve_path,
    )

Pure helpers — no side effects at import time. Each function is small
enough that tests cover the package as a whole rather than each function
individually.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------


def init_logging(verbose: bool = False, *, format_str: str | None = None) -> None:
    """Configure root logging with a consistent SIA format.

    Idempotent — calling twice doesn't double the handlers.
    """
    if logging.getLogger().handlers:
        # Already configured (e.g. by pytest's --log-cli)
        return
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=format_str or "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------


def load_env_or_die(*required_keys: str) -> dict[str, str]:
    """Return a dict of {key: value} for every required env var, or sys.exit(1)
    with a clear message listing the missing ones.

    Always at least checks GROQ_API_KEY when called with no args, since that's
    the most common SIA prerequisite.
    """
    keys = list(required_keys) or ["GROQ_API_KEY"]
    missing: list[str] = []
    found: dict[str, str] = {}
    for key in keys:
        value = os.environ.get(key, "").strip()
        if not value:
            missing.append(key)
        else:
            found[key] = value

    if missing:
        print(
            "Missing required environment variables: "
            + ", ".join(missing)
            + "\nSet them in your .env or shell before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)
    return found


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------


def resolve_path(path: str | Path, *, must_exist: bool = False) -> Path:
    """Expand ``~`` and resolve to an absolute Path.

    Optionally raises FileNotFoundError if ``must_exist`` is set and the
    path doesn't.
    """
    resolved = Path(path).expanduser().resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def ensure_dir(path: str | Path) -> Path:
    """Create the directory (and parents) if missing. Returns the resolved path."""
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------


def make_progress(
    iterable: Iterable[T], total: int | None = None, label: str = ""
) -> Iterator[T]:
    """Wrap ``iterable`` with tqdm if available, else yield unchanged.

    The scripts don't all want the same progress library installed — this
    keeps `import _lib` cheap and falls back gracefully when tqdm isn't
    present.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        # No tqdm installed → just pass through with periodic logger.info
        return _stderr_progress(iterable, total, label)
    return iter(tqdm(iterable, total=total, desc=label or None))


def _stderr_progress(
    iterable: Iterable[T], total: int | None, label: str
) -> Iterator[T]:
    """Minimal fallback: log every N items so the user sees progress."""
    step = max(1, (total or 100) // 20)  # ~20 updates over the run
    for i, item in enumerate(iterable, 1):
        yield item
        if i % step == 0 or i == total:
            if total:
                logger.info("%s %d/%d", label or "progress", i, total)
            else:
                logger.info("%s %d", label or "progress", i)


# ---------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield parsed dicts from a JSONL file. Skips malformed lines with a warn."""
    import json

    resolved = resolve_path(path, must_exist=True)
    with open(resolved) as f:
        for idx, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed line %d in %s: %s", idx, resolved, exc
                )


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows to JSONL, creating parent dirs. Returns count written."""
    import json

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(target, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            count += 1
    return count
