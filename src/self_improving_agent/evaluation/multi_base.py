"""
Multi-base-model evaluation

The eval harness already takes a generate_fn so it's base-model agnostic
at the API level. What's been missing is the workflow: there's no way
to compare Llama 3.1 8B against Qwen2 7B or DeepSeek-Coder-6.7B on the
same suite without writing a custom script for each base.

This module adds:

  - BaseModelSpec: declarative description of one base to evaluate
    (provider, model_path, optional adapter)
  - evaluate_bases(specs, suite_path): runs the harness against each
    base, returns a MultiBaseReport with per-base EvaluationReports
    and a head-to-head comparison view.

Used by the new `sia compare-bases` command (added in commands/compare.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from self_improving_agent.evaluation.harness import EvaluationHarness
from self_improving_agent.evaluation.models import EvaluationReport
from self_improving_agent.providers.base import GenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class BaseModelSpec:
    """One base+adapter combination to evaluate.

    Args:
        label: Human-readable id, used in the report ('llama-3.1-8b' etc).
        provider: 'mlx' or 'groq'.
        model_path: HF id or Groq model name.
        adapter_path: Optional LoRA adapter to load (MLX only).
        base_model: For metadata/reporting. Defaults to model_path.
    """

    label: str
    provider: str
    model_path: str
    adapter_path: str | None = None
    base_model: str | None = None


@dataclass
class MultiBaseReport:
    suite_path: str
    reports: dict[str, EvaluationReport] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        return list(self.reports.keys())

    def winner(self) -> str | None:
        """Label of the base with the highest pass rate (None on tie)."""
        if not self.reports:
            return None
        ranked = sorted(
            self.reports.items(), key=lambda kv: kv[1].pass_rate, reverse=True
        )
        if len(ranked) >= 2 and ranked[0][1].pass_rate == ranked[1][1].pass_rate:
            return None
        return ranked[0][0]

    def comparison_table(self) -> list[dict[str, Any]]:
        """Side-by-side rows suitable for a dashboard or console table."""
        rows = []
        for label, report in self.reports.items():
            rows.append(
                {
                    "label": label,
                    "base_model": report.base_model,
                    "pass_rate": report.pass_rate,
                    "passed": report.passed_tests,
                    "total": report.total_tests,
                    "avg_latency_ms": report.avg_latency_ms,
                }
            )
        return rows

    def category_table(self) -> dict[str, dict[str, float]]:
        """{category: {label: score}} for the dashboard heatmap view."""
        all_categories: set[str] = set()
        for report in self.reports.values():
            all_categories.update(report.category_scores.keys())

        out: dict[str, dict[str, float]] = {}
        for cat in sorted(all_categories):
            row: dict[str, float] = {}
            for label, report in self.reports.items():
                if cat in report.category_scores:
                    row[label] = report.category_scores[cat]
            out[cat] = row
        return out


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------


async def evaluate_bases(
    specs: list[BaseModelSpec],
    suite_path: str | Path | None = None,
    *,
    provider_factory=None,
) -> MultiBaseReport:
    """Evaluate every spec in ``specs`` against the same suite.

    Args:
        specs: list of BaseModelSpec to compare.
        suite_path: path to golden_tests directory (None = bundled suite).
        provider_factory: optional callable returning a provider given a
            spec; defaults to the built-in MLX/Groq dispatcher. Tests
            pass a stub here.

    Returns:
        MultiBaseReport with one EvaluationReport per spec.label.
    """
    if not specs:
        raise ValueError("evaluate_bases needs at least one spec")

    harness = EvaluationHarness(suite_path)
    if not harness.tests:
        raise ValueError(
            f"No golden tests at {harness.test_suite_path}. "
            "Cannot evaluate bases without a corpus."
        )

    factory = provider_factory or _default_provider_factory
    report = MultiBaseReport(suite_path=str(harness.test_suite_path))

    for spec in specs:
        provider = factory(spec)

        async def generate_fn(message, history=None, _provider=provider):
            cfg = GenerationConfig(temperature=0.0, max_tokens=256)
            history = history or []
            messages = [*history, {"role": "user", "content": message}]
            r = await _provider.generate(messages, cfg)
            return r.text

        result = await harness.run_all(
            generate_fn=generate_fn,
            model_version=spec.label,
        )
        result.base_model = spec.base_model or spec.model_path
        report.reports[spec.label] = result
        logger.info(
            "%s: %d/%d (%.0f%%)",
            spec.label,
            result.passed_tests,
            result.total_tests,
            result.pass_rate * 100,
        )

    return report


def _default_provider_factory(spec: BaseModelSpec):
    if spec.provider == "groq":
        from self_improving_agent.providers.groq_provider import GroqProvider

        return GroqProvider(model=spec.model_path)
    if spec.provider == "mlx":
        from self_improving_agent.providers.mlx_provider import MLXProvider

        return MLXProvider(
            model_path=spec.model_path,
            adapter_path=spec.adapter_path,
        )
    raise ValueError(f"Unknown provider {spec.provider!r}")
