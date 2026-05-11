"""CLI entry point for the Self-Improving Agent.

Subcommands:
    status    Show registry summary + last eval per version
    evaluate  Run the golden-test harness against a model version
    train     Run the improvement-cycle pipeline
    serve     Start the inference HTTP server
"""

from __future__ import annotations

import argparse
import logging
import sys

from self_improving_agent.commands import (
    dashboard,
    eval_validate,
    evaluate,
    merge,
    serve,
    status,
    train,
)

__version__ = "0.1.0"


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sia",
        description="Self-Improving Agent — OH AI infrastructure",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    status_p = subparsers.add_parser(
        "status", help="Show registry summary and last eval per version"
    )
    status.add_arguments(status_p)
    status_p.set_defaults(func=status.run)

    evaluate_p = subparsers.add_parser(
        "evaluate", help="Run the golden-test harness against a version"
    )
    evaluate.add_arguments(evaluate_p)
    evaluate_p.set_defaults(func=evaluate.run)

    train_p = subparsers.add_parser(
        "train", help="Run the self-improvement cycle"
    )
    train.add_arguments(train_p)
    train_p.set_defaults(func=train.run)

    serve_p = subparsers.add_parser("serve", help="Start the inference HTTP server")
    serve.add_arguments(serve_p)
    serve_p.set_defaults(func=serve.run)

    eval_validate_p = subparsers.add_parser(
        "eval-validate",
        help="Sanity-check the eval pipeline (used in CI; no inference required)",
    )
    eval_validate.add_arguments(eval_validate_p)
    eval_validate_p.set_defaults(func=eval_validate.run)

    dashboard_p = subparsers.add_parser(
        "dashboard",
        help="Launch the Streamlit eval dashboard",
    )
    dashboard.add_arguments(dashboard_p)
    dashboard_p.set_defaults(func=dashboard.run)

    merge_p = subparsers.add_parser(
        "merge",
        help="Weighted-sum merge of N LoRA adapters into a new adapter",
    )
    merge.add_arguments(merge_p)
    merge_p.set_defaults(func=merge.run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
