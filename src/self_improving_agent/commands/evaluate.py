"""``sia evaluate`` — run the golden-test harness against a model version.

Loads the registry's adapter for ``--version`` (or the current production
adapter if unspecified), wires it through MLXProvider, runs the
EvaluationHarness, and writes a JSON report.

Designed to be re-run cheaply — adapter is loaded once, kept hot for the
whole suite.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from self_improving_agent.evaluation.harness import EvaluationHarness
from self_improving_agent.training.registry import ModelRegistry, ModelVersion

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--version",
        help="Registry version to evaluate. Defaults to current production.",
    )
    parser.add_argument(
        "--registry",
        default="models/registry.json",
        help="Path to registry.json (default: models/registry.json)",
    )
    parser.add_argument(
        "--reports-dir",
        default="evaluation/reports",
        help="Where to write the JSON report (default: evaluation/reports/)",
    )
    parser.add_argument(
        "--test-suite",
        default=None,
        help="Override golden_tests directory (default: bundled suite).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print summary but don't persist the report.",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Evaluate the remote (Groq) provider instead of the local adapter.",
    )


def run(args: argparse.Namespace) -> int:
    registry = ModelRegistry(args.registry)
    target = _resolve_target(registry, args)
    if target is None and not args.remote:
        print(
            "No target model. Specify --version, ensure a production model "
            "is registered, or pass --remote to evaluate Groq."
        )
        return 2

    generate_fn = _build_generate_fn(target, args.remote)

    harness = EvaluationHarness(args.test_suite)
    if not harness.tests:
        print(
            f"No golden tests found at {harness.test_suite_path}. "
            "Add YAML files under that directory."
        )
        return 2

    print(
        f"Evaluating {target.version if target else 'remote'} "
        f"against {len(harness.tests)} test(s)..."
    )
    started = time.time()
    report = asyncio.run(
        harness.run_all(
            generate_fn=generate_fn,
            model_version=target.version if target else "remote",
        )
    )
    elapsed = time.time() - started

    print()
    print(_render_report(report, elapsed))

    if args.no_write:
        return 0 if report.pass_rate >= 0.0 else 1

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    version_tag = report.model_version.replace("/", "_")
    out_path = reports_dir / f"{version_tag}_{int(started)}.json"
    out_path.write_text(json.dumps(_report_to_dict(report), indent=2))
    print(f"\nReport saved to: {out_path}")
    return 0


def _resolve_target(
    registry: ModelRegistry, args: argparse.Namespace
) -> ModelVersion | None:
    if args.remote:
        return None
    if args.version:
        target = registry.get(args.version)
        if target is None:
            raise SystemExit(f"Version '{args.version}' not in registry.")
        return target
    return registry.get_production()


def _build_generate_fn(target: ModelVersion | None, remote: bool):
    """Construct an async generate function for the harness.

    The harness calls ``generate(messages, temperature=, max_tokens=)``. We
    wrap MLXProvider or GroqProvider in a shim with that signature.
    """
    # Local imports keep the import cost off `sia status` / `sia --help`.
    from self_improving_agent.providers.base import GenerationConfig

    if remote or target is None:
        from self_improving_agent.providers.groq_provider import GroqProvider

        provider = GroqProvider()
    else:
        from self_improving_agent.providers.mlx_provider import MLXProvider

        provider = MLXProvider(
            adapter_path=target.adapter_path,
            base_model=target.base_model,
        )

    async def generate_fn(messages, *, temperature: float = 0.0, max_tokens: int = 256):
        config = GenerationConfig(temperature=temperature, max_tokens=max_tokens)
        response = await provider.generate(messages, config)
        return response.text

    return generate_fn


def _render_report(report, elapsed_s: float) -> str:
    lines = [
        f"Pass rate: {report.pass_rate:.0%}  "
        f"({report.passed_tests}/{report.total_tests})",
        f"Mean latency: {report.avg_latency_ms:.0f}ms  Wall time: {elapsed_s:.1f}s",
    ]
    if report.category_scores:
        lines.append("By category:")
        for cat, score in sorted(report.category_scores.items()):
            lines.append(f"  {cat:<24} {score:.0%}")
    failed = [r for r in report.results if not r.passed]
    if failed:
        lines.append(f"\nFailed: {len(failed)}")
        for r in failed[:5]:
            head = r.failures[0] if r.failures else "(no detail)"
            lines.append(f"  - {r.test_id}: {head[:90]}")
        if len(failed) > 5:
            lines.append(f"  ... and {len(failed) - 5} more")
    return "\n".join(lines)


def _report_to_dict(report) -> dict:
    return {
        "model_version": report.model_version,
        "timestamp": report.timestamp,
        "total_tests": report.total_tests,
        "passed_tests": report.passed_tests,
        "failed_tests": report.failed_tests,
        "pass_rate": report.pass_rate,
        "avg_latency_ms": report.avg_latency_ms,
        "category_scores": report.category_scores,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "score": r.score,
                "latency_ms": r.latency_ms,
                "failures": r.failures,
            }
            for r in report.results
        ],
    }
