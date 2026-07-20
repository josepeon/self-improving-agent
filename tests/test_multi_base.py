"""T2.5: multi-base-model evaluation."""

from __future__ import annotations

import pytest

from self_improving_agent.cli import main
from self_improving_agent.commands.compare import _parse_spec
from self_improving_agent.evaluation.multi_base import (
    BaseModelSpec,
    MultiBaseReport,
    evaluate_bases,
)
from self_improving_agent.providers.base import ProviderResponse


class StubProvider:
    """Returns the configured response text for every generate call."""

    def __init__(self, text: str = "hello"):
        self.text = text

    async def generate(self, messages, config):
        return ProviderResponse(
            text=self.text,
            provider="stub",
            model="stub-model",
            latency_ms=0.0,
            tokens_used=2,
        )

    async def generate_stream(self, *_a, **_kw):  # pragma: no cover
        if False:
            yield ""

    async def health_check(self) -> bool:  # pragma: no cover
        return True


@pytest.fixture
def golden_tests(tmp_path):
    """Two trivial golden tests with simple must_contain rubrics."""
    suite = tmp_path / "tests"
    suite.mkdir()
    (suite / "core.yaml").write_text(
        "tests:\n"
        "  - id: t1\n"
        "    messages:\n"
        "      - role: user\n"
        "        content: hi\n"
        "    expected:\n"
        "      must_contain: [hello]\n"
        "  - id: t2\n"
        "    messages:\n"
        "      - role: user\n"
        "        content: greet\n"
        "    expected:\n"
        "      must_contain: [hello]\n"
    )
    return suite


# ---------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------


class TestParseSpec:
    def test_simple_spec(self):
        spec = _parse_spec("llama-8b=mlx/mlx-community/Meta-Llama-3.1-8B")
        assert spec.label == "llama-8b"
        assert spec.provider == "mlx"
        assert spec.model_path == "mlx-community/Meta-Llama-3.1-8B"
        assert spec.adapter_path is None

    def test_spec_with_adapter(self):
        spec = _parse_spec("v1.2=mlx/base/llama+adapter=models/finetuned/v1.2/adapters")
        assert spec.adapter_path == "models/finetuned/v1.2/adapters"
        assert spec.model_path == "base/llama"

    def test_groq_spec(self):
        spec = _parse_spec("groq-70b=groq/llama-3.3-70b-versatile")
        assert spec.provider == "groq"

    def test_missing_equals_rejected(self):
        with pytest.raises(ValueError, match="Spec must include '='"):
            _parse_spec("no-equals-here")

    def test_empty_label_rejected(self):
        with pytest.raises(ValueError, match="Empty label"):
            _parse_spec("=mlx/x")

    def test_missing_slash_rejected(self):
        with pytest.raises(ValueError, match="provider/model"):
            _parse_spec("label=just-model")


# ---------------------------------------------------------------------
# MultiBaseReport helpers
# ---------------------------------------------------------------------


class TestMultiBaseReport:
    def _populate(self, **pass_rates):
        from self_improving_agent.evaluation.models import EvaluationReport

        report = MultiBaseReport(suite_path="x")
        for label, rate in pass_rates.items():
            report.reports[label] = EvaluationReport(
                model_version=label,
                pass_rate=rate,
                passed_tests=int(rate * 10),
                failed_tests=10 - int(rate * 10),
                total_tests=10,
                category_scores={"factual": rate, "tone": rate * 0.9},
            )
        return report

    def test_winner_clear_winner(self):
        report = self._populate(a=0.6, b=0.8, c=0.4)
        assert report.winner() == "b"

    def test_winner_tie_returns_none(self):
        report = self._populate(a=0.8, b=0.8)
        assert report.winner() is None

    def test_winner_empty(self):
        assert MultiBaseReport(suite_path="").winner() is None

    def test_comparison_table_includes_each_label(self):
        report = self._populate(a=0.5, b=0.7)
        rows = report.comparison_table()
        labels = {r["label"] for r in rows}
        assert labels == {"a", "b"}

    def test_category_table_groups_by_category(self):
        report = self._populate(a=0.5, b=0.7)
        cats = report.category_table()
        assert "factual" in cats
        assert cats["factual"]["a"] == 0.5
        assert cats["factual"]["b"] == 0.7


# ---------------------------------------------------------------------
# evaluate_bases end-to-end
# ---------------------------------------------------------------------


class TestEvaluateBases:
    @pytest.mark.asyncio
    async def test_compares_two_specs(self, golden_tests):
        specs = [
            BaseModelSpec(label="hot", provider="stub", model_path="m"),
            BaseModelSpec(label="cold", provider="stub", model_path="m"),
        ]

        # "hot" says 'hello' (passes), "cold" says 'wrong' (fails)
        def factory(spec: BaseModelSpec):
            return StubProvider(text="hello" if spec.label == "hot" else "wrong")

        report = await evaluate_bases(specs, golden_tests, provider_factory=factory)
        assert set(report.labels) == {"hot", "cold"}
        assert report.reports["hot"].pass_rate == 1.0
        assert report.reports["cold"].pass_rate == 0.0
        assert report.winner() == "hot"

    @pytest.mark.asyncio
    async def test_empty_specs_rejected(self, golden_tests):
        with pytest.raises(ValueError, match="at least one"):
            await evaluate_bases([], golden_tests)

    @pytest.mark.asyncio
    async def test_no_golden_tests_raises(self, tmp_path):
        specs = [BaseModelSpec(label="x", provider="stub", model_path="m")]
        with pytest.raises(ValueError, match="Cannot evaluate"):
            await evaluate_bases(
                specs,
                tmp_path / "no-tests-here",
                provider_factory=lambda s: StubProvider(),
            )

    @pytest.mark.asyncio
    async def test_report_includes_base_model(self, golden_tests):
        specs = [
            BaseModelSpec(
                label="llama",
                provider="stub",
                model_path="meta/llama-3-8b",
                base_model="meta/llama-3-8b",
            ),
        ]
        report = await evaluate_bases(
            specs, golden_tests, provider_factory=lambda s: StubProvider(text="hello")
        )
        assert report.reports["llama"].base_model == "meta/llama-3-8b"


# ---------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------


class TestCompareCommand:
    def test_invalid_spec_returns_2(self, tmp_path):
        result = main(
            [
                "compare-bases",
                "--spec",
                "totally-invalid-spec",
                "--test-suite",
                str(tmp_path),
            ]
        )
        assert result == 2
