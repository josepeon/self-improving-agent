"""T1.3: multi-judge ensemble for self-play conversation filtering."""

from __future__ import annotations

import asyncio

import pytest

from self_improving_agent.providers.base import ProviderResponse
from self_improving_agent.training.judges import (
    AGGREGATORS,
    EnsembleJudge,
    JudgeVerdict,
    ProviderJudge,
    _parse_score,
)


# ---------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------


class FakeProvider:
    def __init__(self, text: str = "8", *, name: str = "fake"):
        self._text = text
        self.name = name
        self.calls = 0

    async def generate(self, messages, config):
        self.calls += 1
        return ProviderResponse(
            text=self._text,
            provider=self.name,
            model="fake-model",
            latency_ms=0.0,
            tokens_used=2,
        )

    async def generate_stream(self, *_a, **_kw):  # pragma: no cover
        if False:
            yield ""

    async def health_check(self) -> bool:  # pragma: no cover
        return True


class FailingProvider(FakeProvider):
    async def generate(self, messages, config):
        raise RuntimeError("network down")


# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------


class TestParseScore:
    def test_simple_integer(self):
        assert _parse_score("7") == 7.0

    def test_decimal(self):
        assert _parse_score("8.5") == 8.5

    def test_with_rationale(self):
        # First number wins
        assert _parse_score("9 — well-structured response") == 9.0

    def test_clamps_high(self):
        assert _parse_score("42") == 10.0

    def test_clamps_low_negative_treats_as_5(self):
        # The regex strips the minus sign, so "-5" parses as "5"
        assert _parse_score("-5") == 5.0

    def test_no_number_returns_zero(self):
        assert _parse_score("not a score") == 0.0


# ---------------------------------------------------------------------
# ProviderJudge
# ---------------------------------------------------------------------


class TestProviderJudge:
    @pytest.mark.asyncio
    async def test_scores_via_provider(self):
        judge = ProviderJudge(FakeProvider(text="9"))
        verdict = await judge.score(
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hey"}]
        )
        assert isinstance(verdict, JudgeVerdict)
        assert verdict.score == 9.0
        assert verdict.judge == "fake-judge"
        assert verdict.error is None

    @pytest.mark.asyncio
    async def test_provider_failure_returns_zero_verdict(self):
        judge = ProviderJudge(FailingProvider())
        verdict = await judge.score([{"role": "user", "content": "Hi"}])
        assert verdict.score == 0.0
        assert verdict.error is not None
        assert "RuntimeError" in verdict.error

    @pytest.mark.asyncio
    async def test_calibration_tracks_scores(self):
        provider = FakeProvider(text="6")
        judge = ProviderJudge(provider, name="cal-test")
        for _ in range(3):
            await judge.score([{"role": "user", "content": "x"}])
        assert judge.scores_seen == [6.0, 6.0, 6.0]
        assert judge.mean_score == 6.0

    @pytest.mark.asyncio
    async def test_unparseable_response_yields_zero(self):
        judge = ProviderJudge(FakeProvider(text="absolutely fine"))
        verdict = await judge.score([{"role": "user", "content": "x"}])
        assert verdict.score == 0.0


# ---------------------------------------------------------------------
# EnsembleJudge
# ---------------------------------------------------------------------


class TestEnsembleJudge:
    @pytest.mark.asyncio
    async def test_median_aggregation(self):
        ensemble = EnsembleJudge(
            judges=[
                ProviderJudge(FakeProvider(text="6"), name="a"),
                ProviderJudge(FakeProvider(text="8"), name="b"),
                ProviderJudge(FakeProvider(text="9"), name="c"),
            ],
            aggregation="median",
        )
        verdict = await ensemble.score([{"role": "user", "content": "x"}])
        assert verdict.score == 8.0
        assert len(verdict.judges) == 3

    @pytest.mark.asyncio
    async def test_mean_aggregation(self):
        ensemble = EnsembleJudge(
            judges=[
                ProviderJudge(FakeProvider(text="6"), name="a"),
                ProviderJudge(FakeProvider(text="8"), name="b"),
            ],
            aggregation="mean",
        )
        verdict = await ensemble.score([{"role": "user", "content": "x"}])
        assert verdict.score == 7.0

    @pytest.mark.asyncio
    async def test_min_aggregation(self):
        ensemble = EnsembleJudge(
            judges=[
                ProviderJudge(FakeProvider(text="3"), name="strict"),
                ProviderJudge(FakeProvider(text="9"), name="lax"),
            ],
            aggregation="min",
        )
        verdict = await ensemble.score([{"role": "user", "content": "x"}])
        # 'min' picks the strict judge's score
        assert verdict.score == 3.0

    @pytest.mark.asyncio
    async def test_failed_judge_excluded_from_aggregate(self):
        ensemble = EnsembleJudge(
            judges=[
                ProviderJudge(FakeProvider(text="9"), name="ok"),
                ProviderJudge(FailingProvider(), name="dead"),
            ],
            aggregation="median",
        )
        verdict = await ensemble.score([{"role": "user", "content": "x"}])
        assert verdict.score == 9.0
        # Both verdicts present; the failing one has error set
        assert len(verdict.judges) == 2
        assert any(v.error for v in verdict.judges)

    @pytest.mark.asyncio
    async def test_is_kept_respects_threshold(self):
        ensemble = EnsembleJudge(
            judges=[ProviderJudge(FakeProvider(text="7"))],
            threshold=7.0,
        )
        verdict = await ensemble.score([{"role": "user", "content": "x"}])
        assert ensemble.is_kept(verdict) is True

        ensemble2 = EnsembleJudge(
            judges=[ProviderJudge(FakeProvider(text="5"))],
            threshold=7.0,
        )
        verdict2 = await ensemble2.score([{"role": "user", "content": "x"}])
        assert ensemble2.is_kept(verdict2) is False

    def test_unknown_aggregation_rejected(self):
        with pytest.raises(ValueError, match="Unknown aggregation"):
            EnsembleJudge(
                judges=[ProviderJudge(FakeProvider())],
                aggregation="weighted_geometric_mean",
            )

    def test_empty_ensemble_rejected(self):
        with pytest.raises(ValueError, match="at least one judge"):
            EnsembleJudge(judges=[])

    @pytest.mark.asyncio
    async def test_judges_run_in_parallel(self):
        # All judges take ~50ms; ensemble should finish in roughly that,
        # not N × 50ms.
        class SlowProvider(FakeProvider):
            async def generate(self, messages, config):
                await asyncio.sleep(0.05)
                return await super().generate(messages, config)

        ensemble = EnsembleJudge(
            judges=[
                ProviderJudge(SlowProvider(text="8"), name=f"j{i}")
                for i in range(4)
            ],
            max_parallel=4,
        )
        import time

        start = time.perf_counter()
        await ensemble.score([{"role": "user", "content": "x"}])
        elapsed = time.perf_counter() - start
        # Should be ~50ms, definitely not 200ms
        assert elapsed < 0.15

    @pytest.mark.asyncio
    async def test_calibration_includes_per_judge_stats(self):
        ensemble = EnsembleJudge(
            judges=[
                ProviderJudge(FakeProvider(text="6"), name="a"),
                ProviderJudge(FakeProvider(text="8"), name="b"),
            ],
        )
        await ensemble.score([{"role": "user", "content": "x"}])
        await ensemble.score([{"role": "user", "content": "x"}])
        cal = ensemble.calibration()
        assert cal["a"]["count"] == 2
        assert cal["a"]["mean"] == 6.0
        assert cal["b"]["mean"] == 8.0


# ---------------------------------------------------------------------
# Aggregator registry sanity
# ---------------------------------------------------------------------


class TestAggregatorRegistry:
    def test_known_aggregators(self):
        assert set(AGGREGATORS.keys()) == {"median", "mean", "min"}

    def test_empty_scores_all_return_zero(self):
        for agg in AGGREGATORS.values():
            assert agg([]) == 0.0
