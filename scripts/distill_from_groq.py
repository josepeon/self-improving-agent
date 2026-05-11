#!/usr/bin/env python3
"""Distill Groq 70B → local SFT dataset.

Reads a JSONL of {"prompt": "..."} rows, asks Groq for each, filters via
the configured judge, and writes SFT JSONL that the existing
dataset_builder / train_oh_model_* scripts can consume.

Usage:
    python scripts/distill_from_groq.py \\
        --input data/questions.jsonl \\
        --output data/training/distilled_v1.jsonl \\
        --threshold 7.0 \\
        --concurrent 4
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Project root on path so 'from self_improving_agent...' works when invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from self_improving_agent.observability.cost_tracker import (  # noqa: E402
    begin_run,
    render_summary,
)
from self_improving_agent.providers.groq_provider import GroqProvider  # noqa: E402
from self_improving_agent.reproducibility import (  # noqa: E402
    get_seed_from_env,
    set_seed,
)
from self_improving_agent.training.distillation import (  # noqa: E402
    JsonlQuestionPool,
    run_distillation,
)
from self_improving_agent.training.judges import ProviderJudge  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSONL file of {prompt: ...} rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write SFT JSONL.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=7.0,
        help="Judge score threshold (0-10) for keeping a pair (default: 7.0).",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=4,
        help="Max concurrent teacher+judge calls (default: 4).",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt prepended to every teacher call.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: $SIA_SEED or 42).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Debug logging."
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    set_seed(args.seed if args.seed is not None else get_seed_from_env())
    begin_run()

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    pool = JsonlQuestionPool(args.input)
    teacher = GroqProvider()
    judges = [
        ProviderJudge(GroqProvider(), name="groq-strict"),
    ]

    report = await run_distillation(
        pool=pool,
        output_path=args.output,
        teacher=teacher,
        judges=judges,
        judge_threshold=args.threshold,
        system_prompt=args.system_prompt,
        max_concurrent=args.concurrent,
    )

    print(report.render())
    print()
    print(render_summary())
    return 0 if report.pairs_kept > 0 else 2


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
