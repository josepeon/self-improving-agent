"""``sia status`` — show registry summary and last evaluation per version.

Output is intentionally compact and grep-friendly. Designed to answer:
  - Which version is currently in production?
  - When did each version last get evaluated?
  - What's the headline pass rate for each version?

Reads only — never mutates state.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from self_improving_agent.training.registry import ModelRegistry, ModelVersion

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the subparser arguments."""
    parser.add_argument(
        "--registry",
        default="models/registry.json",
        help="Path to registry.json (default: models/registry.json)",
    )
    parser.add_argument(
        "--reports-dir",
        default="evaluation/reports",
        help="Where evaluation reports live (default: evaluation/reports)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON for machine consumption instead of the table.",
    )


def run(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry)
    if not registry_path.exists():
        print(f"No registry at {registry_path}. Nothing to report yet.")
        return 0

    registry = ModelRegistry(registry_path)
    versions = sorted(
        registry.list_versions(),
        key=lambda v: v.created_at,
        reverse=True,
    )

    reports_dir = Path(args.reports_dir)
    enriched = [_attach_latest_report(v, reports_dir) for v in versions]

    if args.json:
        print(json.dumps(enriched, indent=2, default=str))
        return 0

    _render_table(enriched)
    return 0


def _attach_latest_report(version: ModelVersion, reports_dir: Path) -> dict:
    """Best-effort lookup of the latest evaluation report for ``version``."""
    report: dict | None = None
    if reports_dir.exists():
        # Look for files like reports/<version>_*.json or reports/<version>.json
        candidates = sorted(
            (p for p in reports_dir.glob(f"{version.version}*.json") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            try:
                with open(candidates[0]) as f:
                    report = json.load(f)
            except (OSError, json.JSONDecodeError):
                report = None

    row = version.to_dict()
    if report:
        row["latest_report"] = {
            "timestamp": report.get("timestamp"),
            "pass_rate": report.get("pass_rate"),
            "passed_tests": report.get("passed_tests"),
            "total_tests": report.get("total_tests"),
            "avg_latency_ms": report.get("avg_latency_ms"),
        }
    return row


def _render_table(rows: list[dict]) -> None:
    if not rows:
        print("Registry is empty. Train a version to see status.")
        return

    headers = ["VERSION", "STATUS", "BASE", "EVAL", "PASS", "CREATED"]
    widths = [
        max(len(headers[0]), max(len(str(r["version"])) for r in rows)),
        max(len(headers[1]), max(len(str(r["status"])) for r in rows)),
        max(len(headers[2]), max(len(str(r["base_model"])[:24]) for r in rows)),
        len(headers[3]),
        len(headers[4]),
        len(headers[5]),
    ]

    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))

    for row in rows:
        report = row.get("latest_report")
        eval_marker = "—"
        pass_str = "—"
        if report and report.get("pass_rate") is not None:
            eval_marker = "✓"
            pass_str = f"{report['pass_rate']:.0%}"
        elif row.get("evaluation_pass_rate") is not None:
            pass_str = f"{row['evaluation_pass_rate']:.0%}"

        created = (row.get("created_at") or "")[:10]
        print(
            fmt.format(
                row["version"],
                row["status"],
                str(row["base_model"])[:24],
                eval_marker,
                pass_str,
                created,
            )
        )

    prod = next((r for r in rows if r["status"] == "production"), None)
    if prod:
        print(f"\nProduction: {prod['version']} (base: {prod['base_model']})")
    else:
        print("\nProduction: <none registered>")
