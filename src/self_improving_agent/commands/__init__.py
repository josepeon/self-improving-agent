"""CLI subcommand implementations.

Each module exposes a ``run(args)`` callable returning an exit code.
``cli.main`` wires them up to argparse subparsers.
"""

from self_improving_agent.commands import (
    compare,
    dashboard,
    eval_validate,
    evaluate,
    merge,
    serve,
    status,
    train,
)

__all__ = [
    "compare",
    "dashboard",
    "eval_validate",
    "evaluate",
    "merge",
    "serve",
    "status",
    "train",
]
