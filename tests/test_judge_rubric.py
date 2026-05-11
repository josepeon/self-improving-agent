"""T3.4: LLM-as-judge rubric in the evaluation harness."""

from __future__ import annotations

import pytest

from self_improving_agent.evaluation.harness import (
    EvaluationHarness,
    _parse_judge_score,
    clear_judge_cache,
)
from self_improving_agent.providers.base import ProviderResponse


@pytest.fixture(autouse=True)
def _cleanup_cache():
    clear_judge_cache()
    yield
    clear_judge_cache()


# ─────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────


class TestParseJudgeScore:
    def test_simple(self):
        assert _parse_judge_score("0.8") == 0.8

    def test_integer_one(self):
        assert _parse_judge_score("1") == 1.0

    def test_with_rationale(self):
        assert _parse_judge_score("0.65 — close to the bar") == 0.65

    def test_normalises_0_to_10_scale(self):
        # Some judges produce 0..10 by habit; we normalise.
        assert _parse_judge_score("8") == 0.8

    def test_unparseable(self):
        assert _parse_judge_score("no number") is None

    def test_clamps(self):
        assert _parse_judge_score("-1") == 1.0  # regex strips minus, "1" -> 1.0


# ─────────────────────────────────────────────────────────────────────
# Judge integration with the harness
# ─────────────────────────────────────────────────────────────────────


class FakeJudgeProvider:
    """Stub judge — returns a configurable score."""

    def __init__(self, text: str = "0.9"):
        self._text = text
        self.calls = 0

    async def generate(self, messages, config):
        self.calls += 1
        return ProviderResponse(
            text=self._text,
            provider="fake-judge",
            model="fake",
            latency_ms=0.0,
            tokens_used=2,
        )


def _patch_judge(monkeypatch, fake):
    """Make the harness use FakeJudgeProvider instead of touching Groq/MLX."""
    from self_improving_agent.evaluation import harness as harness_module

    monkeypatch.setattr(
        harness_module,
        "_get_judge_provider",
        lambda *_a, **_kw: fake,
    )


def _make_test(criterion: str, threshold: float = 0.7) -> dict:
    return {
        "id": "judged",
        "messages": [{"role": "user", "content": "Hello"}],
        "expected": {
            "judge": {
                "model": "groq",
                "criterion": criterion,
                "threshold": threshold,
            }
        },
    }


class TestJudgeRubric:
    @pytest.mark.asyncio
    async def test_passing_score_no_failure(self, tmp_path, monkeypatch):
        fake = FakeJudgeProvider(text="0.95")
        _patch_judge(monkeypatch, fake)

        async def generate_fn(message, history=None):
            return "Hi there, glad you stopped by!"

        harness = EvaluationHarness(tmp_path)  # empty test dir
        result = await harness.run_test(_make_test("friendly tone"), generate_fn)
        assert result.passed is True
        assert result.failures == []
        assert fake.calls == 1

    @pytest.mark.asyncio
    async def test_failing_score_records_failure(self, tmp_path, monkeypatch):
        fake = FakeJudgeProvider(text="0.4")
        _patch_judge(monkeypatch, fake)

        async def generate_fn(message, history=None):
            return "..."

        harness = EvaluationHarness(tmp_path)
        result = await harness.run_test(_make_test("friendly tone"), generate_fn)
        assert result.passed is False
        assert any("Judge score" in f for f in result.failures)

    @pytest.mark.asyncio
    async def test_unparseable_judge_records_failure(self, tmp_path, monkeypatch):
        fake = FakeJudgeProvider(text="not a number")
        _patch_judge(monkeypatch, fake)

        async def generate_fn(message, history=None):
            return "..."

        harness = EvaluationHarness(tmp_path)
        result = await harness.run_test(_make_test("friendly tone"), generate_fn)
        assert any("unparseable" in f for f in result.failures)

    @pytest.mark.asyncio
    async def test_judge_provider_failure_records_failure(self, tmp_path, monkeypatch):
        class BoomProvider:
            async def generate(self, *a, **kw):
                raise RuntimeError("network")

        _patch_judge(monkeypatch, BoomProvider())

        async def generate_fn(message, history=None):
            return "..."

        harness = EvaluationHarness(tmp_path)
        result = await harness.run_test(_make_test("anything"), generate_fn)
        assert any("Judge unavailable" in f for f in result.failures)

    @pytest.mark.asyncio
    async def test_no_judge_means_no_judge_call(self, tmp_path, monkeypatch):
        fake = FakeJudgeProvider(text="0.9")
        _patch_judge(monkeypatch, fake)

        async def generate_fn(message, history=None):
            return "Hello"

        plain_test = {
            "id": "plain",
            "messages": [{"role": "user", "content": "Hi"}],
            "expected": {"must_contain": ["Hello"]},
        }

        harness = EvaluationHarness(tmp_path)
        result = await harness.run_test(plain_test, generate_fn)
        assert result.passed is True
        # Crucially, the judge wasn't called at all
        assert fake.calls == 0

    @pytest.mark.asyncio
    async def test_missing_criterion_records_failure(self, tmp_path, monkeypatch):
        fake = FakeJudgeProvider(text="0.9")
        _patch_judge(monkeypatch, fake)

        async def generate_fn(message, history=None):
            return "anything"

        bad_test = {
            "id": "bad",
            "messages": [{"role": "user", "content": "Hi"}],
            "expected": {"judge": {"model": "groq"}},  # no criterion
        }

        harness = EvaluationHarness(tmp_path)
        result = await harness.run_test(bad_test, generate_fn)
        assert any("missing 'criterion'" in f for f in result.failures)

    @pytest.mark.asyncio
    async def test_unknown_provider_records_failure(self, tmp_path, monkeypatch):
        # Don't patch; let the real factory choke on the unknown name
        async def generate_fn(message, history=None):
            return "x"

        bad_test = {
            "id": "bad",
            "messages": [{"role": "user", "content": "Hi"}],
            "expected": {
                "judge": {
                    "model": "madeup-provider",
                    "criterion": "anything",
                }
            },
        }

        harness = EvaluationHarness(tmp_path)
        result = await harness.run_test(bad_test, generate_fn)
        assert any("Could not build judge" in f for f in result.failures)
