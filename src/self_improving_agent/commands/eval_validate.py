"""``sia eval-validate`` — sanity check the eval pipeline without inference.

Runs in CI on every push. Catches "the eval harness broke" without needing
GPU hardware or API keys. Specifically:

  - Every YAML in golden_tests/ parses as a valid test spec
  - models/registry.json (if present) loads without errors
  - Every existing report in evaluation/reports/ has the expected shape
  - There's at least one golden test (catches accidental deletion)

Exits non-zero on any failure with a one-line summary of what's broken.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from self_improving_agent.evaluation.harness import EvaluationHarness
from self_improving_agent.training.registry import ModelRegistry

logger = logging.getLogger(__name__)

REQUIRED_TEST_KEYS = {"id", "messages"}
REQUIRED_REPORT_KEYS = {
    "model_version",
    "total_tests",
    "passed_tests",
    "pass_rate",
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--test-suite",
        default=None,
        help="Override golden_tests directory (default: bundled suite).",
    )
    parser.add_argument(
        "--registry",
        default="models/registry.json",
        help="Path to registry.json (default: models/registry.json)",
    )
    parser.add_argument(
        "--reports-dir",
        default="evaluation/reports",
        help="Path to evaluation reports (default: evaluation/reports)",
    )
    parser.add_argument(
        "--min-tests",
        type=int,
        default=1,
        help="Minimum number of golden tests required (default: 1).",
    )


def run(args: argparse.Namespace) -> int:
    errors: list[str] = []

    # 1) Golden tests parse and have required keys
    harness = EvaluationHarness(args.test_suite)
    if len(harness.tests) < args.min_tests:
        errors.append(
            f"Only {len(harness.tests)} golden test(s) found at "
            f"{harness.test_suite_path}; required minimum is {args.min_tests}."
        )
    for i, test in enumerate(harness.tests):
        missing = REQUIRED_TEST_KEYS - set(test.keys())
        if missing:
            errors.append(
                f"Test #{i} ({test.get('id', '<no id>')}) missing keys: "
                f"{sorted(missing)}"
            )
        # Each test must have at least one message
        if not test.get("messages"):
            errors.append(f"Test #{i} ({test.get('id', '<no id>')}) has empty messages")

    # 2) Registry loads (if present) and every status is valid
    registry_path = Path(args.registry)
    if registry_path.exists():
        try:
            registry = ModelRegistry(registry_path)
            # Touch every entry to force any deserialization errors
            _ = list(registry.versions.values())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Registry at {registry_path} failed to load: {exc}")

    # 3) Existing reports have the right shape
    reports_dir = Path(args.reports_dir)
    if reports_dir.exists():
        for report_path in sorted(reports_dir.glob("*.json")):
            try:
                with open(report_path) as f:
                    report = json.load(f)
            except json.JSONDecodeError as exc:
                errors.append(f"{report_path.name}: not valid JSON ({exc})")
                continue
            missing = REQUIRED_REPORT_KEYS - set(report.keys())
            if missing:
                errors.append(
                    f"{report_path.name}: missing required keys {sorted(missing)}"
                )

    if errors:
        print(f"FAIL: {len(errors)} validation error(s):")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(
        f"OK: {len(harness.tests)} golden test(s), "
        f"registry={'present' if registry_path.exists() else 'absent'}, "
        f"{len(list(reports_dir.glob('*.json'))) if reports_dir.exists() else 0} "
        "report(s)."
    )
    return 0
