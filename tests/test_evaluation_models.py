"""Tests for evaluation models - TestResult, EvaluationReport."""

from __future__ import annotations

from self_improving_agent.evaluation.models import EvaluationReport, TestResult

# ── TestResult ────────────────────────────────────────────────────────


class TestTestResult:
    def test_defaults(self) -> None:
        r = TestResult(
            test_id="t1", passed=True, score=1.0, latency_ms=50, response="ok"
        )
        assert r.failures == []
        assert r.passed

    def test_with_failures(self) -> None:
        r = TestResult(
            test_id="t2",
            passed=False,
            score=0.5,
            latency_ms=100,
            response="bad",
            failures=["missing phrase"],
        )
        assert not r.passed
        assert len(r.failures) == 1


# ── EvaluationReport ─────────────────────────────────────────────────


def _make_report(**overrides: object) -> EvaluationReport:
    defaults: dict = {
        "model_version": "base",
        "total_tests": 10,
        "passed_tests": 9,
        "failed_tests": 1,
        "pass_rate": 0.9,
        "avg_latency_ms": 200.0,
        "category_scores": {"greeting": 1.0, "safety": 0.8},
    }
    defaults.update(overrides)
    return EvaluationReport(**defaults)


class TestEvaluationReport:
    def test_has_timestamp(self) -> None:
        r = _make_report()
        assert r.timestamp  # non-empty ISO string

    def test_summary_contains_version(self) -> None:
        r = _make_report()
        assert "base" in r.summary()

    def test_summary_shows_categories(self) -> None:
        r = _make_report()
        s = r.summary()
        assert "greeting" in s
        assert "safety" in s

    def test_summary_shows_failures(self) -> None:
        result = TestResult(
            test_id="fail_1",
            passed=False,
            score=0.0,
            latency_ms=50,
            response="bad",
            failures=["Missing required phrase: 'OH'"],
        )
        r = _make_report(results=[result])
        assert "fail_1" in r.summary()
        assert "Missing required phrase" in r.summary()


# ── Promotion logic ──────────────────────────────────────────────────


class TestPromotionReady:
    def test_same_stats_is_ready(self) -> None:
        baseline = _make_report()
        candidate = _make_report()
        assert candidate.is_promotion_ready(baseline)

    def test_better_pass_rate_is_ready(self) -> None:
        baseline = _make_report(pass_rate=0.8)
        candidate = _make_report(pass_rate=0.9)
        assert candidate.is_promotion_ready(baseline)

    def test_worse_pass_rate_not_ready(self) -> None:
        baseline = _make_report(pass_rate=0.9)
        candidate = _make_report(pass_rate=0.8)
        assert not candidate.is_promotion_ready(baseline)

    def test_latency_within_20pct_is_ready(self) -> None:
        baseline = _make_report(avg_latency_ms=100.0)
        candidate = _make_report(avg_latency_ms=119.0)
        assert candidate.is_promotion_ready(baseline)

    def test_latency_over_20pct_not_ready(self) -> None:
        baseline = _make_report(avg_latency_ms=100.0)
        candidate = _make_report(avg_latency_ms=121.0)
        assert not candidate.is_promotion_ready(baseline)

    def test_category_drop_under_5pct_is_ready(self) -> None:
        baseline = _make_report(category_scores={"greeting": 1.0})
        candidate = _make_report(category_scores={"greeting": 0.96})
        assert candidate.is_promotion_ready(baseline)

    def test_category_drop_over_5pct_not_ready(self) -> None:
        baseline = _make_report(category_scores={"greeting": 1.0})
        candidate = _make_report(category_scores={"greeting": 0.94})
        assert not candidate.is_promotion_ready(baseline)

    def test_new_category_in_candidate_ok(self) -> None:
        baseline = _make_report(category_scores={"greeting": 1.0})
        candidate = _make_report(category_scores={"greeting": 1.0, "memory": 0.5})
        assert candidate.is_promotion_ready(baseline)

    def test_zero_baseline_latency_ok(self) -> None:
        baseline = _make_report(avg_latency_ms=0.0)
        candidate = _make_report(avg_latency_ms=500.0)
        assert candidate.is_promotion_ready(baseline)
