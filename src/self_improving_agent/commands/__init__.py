"""CLI subcommand implementations.

Each module exposes a ``run(args)`` callable returning an exit code.
``cli.main`` wires them up to argparse subparsers.
"""

from self_improving_agent.commands import (
    eval_validate,
    evaluate,
    serve,
    status,
    train,
)

__all__ = ["eval_validate", "evaluate", "serve", "status", "train"]
