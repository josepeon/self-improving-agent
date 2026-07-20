"""``sia compare-bases`` — head-to-head evaluation of multiple bases.

Spec format: ``label=provider/model[+adapter=PATH]``. Repeatable.

Example:
    sia compare-bases \\
        --spec llama-8b=mlx/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit+adapter=models/finetuned/v1.2-oh/adapters \\
        --spec qwen-7b=mlx/mlx-community/Qwen2-7B-Instruct-4bit \\
        --spec groq-70b=groq/llama-3.3-70b-versatile
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from self_improving_agent.evaluation.multi_base import (
    BaseModelSpec,
    MultiBaseReport,
    evaluate_bases,
)

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--spec",
        action="append",
        required=True,
        help=(
            "Base spec 'label=provider/model[+adapter=PATH]'. Repeatable. "
            "providers: 'mlx', 'groq'."
        ),
    )
    parser.add_argument(
        "--test-suite",
        default=None,
        help="Path to golden_tests dir (default: bundled).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file to write the comparison report to.",
    )


def run(args: argparse.Namespace) -> int:
    try:
        specs = [_parse_spec(s) for s in args.spec]
    except ValueError as exc:
        print(f"Spec error: {exc}")
        return 2

    report = asyncio.run(evaluate_bases(specs, args.test_suite))

    print(_render_table(report))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "suite_path": report.suite_path,
            "comparison": report.comparison_table(),
            "categories": report.category_table(),
            "winner": report.winner(),
        }
        out_path.write_text(json.dumps(data, indent=2))
        print(f"\nReport saved to: {out_path}")

    return 0


def _parse_spec(raw: str) -> BaseModelSpec:
    """Parse 'label=provider/model[+adapter=PATH]'."""
    if "=" not in raw:
        raise ValueError(f"Spec must include '=': {raw!r}")
    label, _, rest = raw.partition("=")
    label = label.strip()
    if not label:
        raise ValueError(f"Empty label in {raw!r}")

    adapter_path: str | None = None
    if "+adapter=" in rest:
        rest, _, adapter_path = rest.partition("+adapter=")

    if "/" not in rest:
        raise ValueError(f"Expected 'provider/model' in {raw!r}; got {rest!r}")
    provider, _, model = rest.partition("/")
    provider = provider.strip()
    model = model.strip()
    if not provider or not model:
        raise ValueError(f"Empty provider or model in {raw!r}")

    return BaseModelSpec(
        label=label,
        provider=provider,
        model_path=model,
        adapter_path=adapter_path.strip() if adapter_path else None,
        base_model=model,
    )


def _render_table(report: MultiBaseReport) -> str:
    rows = report.comparison_table()
    if not rows:
        return "No results."
    width = max(len(r["label"]) for r in rows)
    lines = [f"{'label':<{width}}  pass  passed/total  avg_latency_ms"]
    for r in sorted(rows, key=lambda x: -x["pass_rate"]):
        lines.append(
            f"{r['label']:<{width}}  "
            f"{r['pass_rate']:.0%}    "
            f"{r['passed']}/{r['total']}        "
            f"{r['avg_latency_ms']:.0f}"
        )
    winner = report.winner()
    if winner:
        lines.append(f"\nWinner: {winner}")
    else:
        lines.append("\nWinner: <tie>")
    return "\n".join(lines)
