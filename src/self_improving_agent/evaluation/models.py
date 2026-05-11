"""Evaluation data models — TestResult, EvaluationReport, and golden test schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class TestResult:
    """Outcome of a single golden test case."""

    __test__ = False  # prevent pytest collection

    test_id: str
    passed: bool
    score: float  # 0.0 - 1.0
    latency_ms: float
    response: str
    failures: list[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Aggregate report produced by :class:`EvaluationHarness.run_all`."""

    model_version: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    pass_rate: float = 0.0
    avg_latency_ms: float = 0.0
    category_scores: dict[str, float] = field(default_factory=dict)
    results: list[TestResult] = field(default_factory=list)
    # T2.5: identify the base model so reports comparing different bases
    # (Llama vs Qwen vs DeepSeek) stay separable in the dashboard.
    base_model: str | None = None

    def is_promotion_ready(self, baseline: EvaluationReport) -> bool:
        """Decide whether this version should replace *baseline*.

        Promotion rules
        ----------------
        1. Pass rate must not decrease.
        2. Average latency must not increase by more than 20 %.
        3. No category score may drop by more than 5 %.
        """
        if self.pass_rate < baseline.pass_rate:
            return False
        if baseline.avg_latency_ms > 0 and (
            self.avg_latency_ms > baseline.avg_latency_ms * 1.2
        ):
            return False
        for cat, score in self.category_scores.items():
            baseline_score = baseline.category_scores.get(cat)
            if baseline_score is not None and score < baseline_score * 0.95:
                return False
        return True

    def summary(self) -> str:
        """Human-readable one-paragraph summary."""
        lines = [
            f"Model {self.model_version} — "
            f"{self.passed_tests}/{self.total_tests} passed "
            f"({self.pass_rate:.0%}), "
            f"avg latency {self.avg_latency_ms:.0f} ms",
        ]
        for cat, score in sorted(self.category_scores.items()):
            lines.append(f"  {cat}: {score:.0%}")
        if self.results:
            failed = [r for r in self.results if not r.passed]
            if failed:
                lines.append("Failures:")
                for r in failed:
                    lines.append(f"  {r.test_id}: {', '.join(r.failures)}")
        return "\n".join(lines)
