"""T3.1: cost & token tracking — pricing, recording, run boundary, budgeting."""

from __future__ import annotations

import threading

import pytest

from self_improving_agent.observability.cost_tracker import (
    BudgetExceededError,
    CostTracker,
    ModelPrice,
    attribute,
    begin_run,
    get_price,
    get_tracker,
    record_usage,
    render_summary,
    to_dict,
)


@pytest.fixture(autouse=True)
def _reset_tracker():
    begin_run()
    yield
    begin_run()


class TestPricing:
    def test_known_model_priced(self):
        price = get_price("openai", "gpt-4o")
        assert price is not None
        assert price.input_per_mtok > 0

    def test_groq_priced_at_zero(self):
        price = get_price("groq", "llama-3.3-70b-versatile")
        assert price is not None
        assert price.input_per_mtok == 0.0

    def test_mlx_wildcard(self):
        price = get_price("mlx", "totally-made-up-model")
        assert price is not None
        assert price.input_per_mtok == 0.0

    def test_unknown_unpriced(self):
        assert get_price("madeup", "madeup") is None

    def test_price_math(self):
        p = ModelPrice(input_per_mtok=2.5, output_per_mtok=10.0)
        # 1000 prompt + 1000 completion at $2.5/$10 per million
        assert p.cost(1000, 1000) == pytest.approx((2.5 + 10.0) / 1000)


class TestRecording:
    def test_record_increments_total(self):
        record_usage("openai", "gpt-4o", 100, 50)
        totals = get_tracker().total()
        assert totals.calls == 1
        assert totals.prompt_tokens == 100
        assert totals.completion_tokens == 50
        assert totals.cost_usd > 0

    def test_unknown_model_flagged(self):
        record = record_usage("madeup", "madeup", 100, 50)
        assert record.priced is False
        assert record.cost_usd == 0.0
        assert get_tracker().total().unpriced_calls == 1

    def test_by_model_aggregation(self):
        record_usage("openai", "gpt-4o", 100, 100)
        record_usage("openai", "gpt-4o", 200, 100)
        record_usage("groq", "llama-3.3-70b-versatile", 500, 200)

        by_model = get_tracker().by_model()
        assert by_model["openai/gpt-4o"].calls == 2
        assert by_model["openai/gpt-4o"].prompt_tokens == 300
        assert by_model["groq/llama-3.3-70b-versatile"].prompt_tokens == 500


class TestRunLifecycle:
    def test_begin_run_resets(self):
        record_usage("openai", "gpt-4o", 100, 100)
        assert get_tracker().total().calls == 1
        begin_run()
        assert get_tracker().total().calls == 0

    def test_budget_enforced(self):
        # gpt-4o = $2.5 in / $10 out per mtok
        begin_run(budget_usd=0.01)
        with pytest.raises(BudgetExceededError):
            record_usage("openai", "gpt-4o", 1000, 1000)  # $0.0125 — exceeds $0.01

    def test_budget_not_exceeded(self):
        begin_run(budget_usd=10.0)
        record_usage("openai", "gpt-4o", 100, 100)
        # No exception


class TestReporting:
    def test_summary_no_calls(self):
        assert "0 calls" in render_summary()

    def test_summary_with_calls(self):
        record_usage("openai", "gpt-4o", 1000, 500)
        out = render_summary()
        assert "$" in out
        assert "openai/gpt-4o" not in out  # only 1 model -> no breakdown line

    def test_summary_breakdown_with_multiple_models(self):
        record_usage("openai", "gpt-4o", 100, 50)
        record_usage("groq", "llama-3.3-70b-versatile", 200, 100)
        out = render_summary()
        assert "openai/gpt-4o" in out
        assert "groq/llama-3.3-70b-versatile" in out

    def test_to_dict_snapshot(self):
        record_usage("openai", "gpt-4o", 100, 50)
        snap = to_dict()
        assert snap["total"]["calls"] == 1
        assert "openai/gpt-4o" in snap["by_model"]

    def test_unpriced_flagged_in_summary(self):
        record_usage("madeup", "madeup", 100, 100)
        assert "no rate card" in render_summary()


class TestAttribution:
    """T-cross-repo: per-role attribution via the attribute() context manager."""

    def test_default_role_is_unknown(self):
        rec = record_usage("groq", "llama-3.3-70b-versatile", 10, 5)
        assert rec.agent_role == "unknown"

    def test_attribute_context_sets_role(self):
        with attribute("planner"):
            rec = record_usage("groq", "llama-3.3-70b-versatile", 10, 5)
        assert rec.agent_role == "planner"

    def test_explicit_agent_role_kwarg_wins(self):
        with attribute("planner"):
            rec = record_usage(
                "groq", "llama-3.3-70b-versatile", 10, 5,
                agent_role="explicit-override",
            )
        assert rec.agent_role == "explicit-override"

    def test_nested_attribute_innermost_wins(self):
        with attribute("outer"):
            with attribute("inner"):
                rec = record_usage("groq", "llama-3.3-70b-versatile", 10, 5)
        assert rec.agent_role == "inner"

    def test_context_resets_after_exit(self):
        with attribute("temp-role"):
            pass
        rec = record_usage("groq", "llama-3.3-70b-versatile", 10, 5)
        assert rec.agent_role == "unknown"

    def test_by_agent_breakdown(self):
        with attribute("planner"):
            record_usage("groq", "llama-3.3-70b-versatile", 100, 50)
            record_usage("groq", "llama-3.3-70b-versatile", 100, 50)
        with attribute("developer"):
            record_usage("openai", "gpt-4o", 200, 100)

        by_agent = get_tracker().by_agent()
        assert by_agent["planner"].calls == 2
        assert by_agent["planner"].prompt_tokens == 200
        assert by_agent["developer"].calls == 1
        assert by_agent["developer"].prompt_tokens == 200

    def test_summary_includes_by_agent_block(self):
        with attribute("planner"):
            record_usage("openai", "gpt-4o", 100, 50)
        with attribute("developer"):
            record_usage("openai", "gpt-4o", 200, 100)

        out = render_summary()
        assert "by agent:" in out
        assert "planner" in out
        assert "developer" in out

    def test_summary_skips_by_agent_when_only_unknown(self):
        # No attribute() scope -> only "unknown" role -> redundant breakdown
        record_usage("openai", "gpt-4o", 100, 50)
        out = render_summary()
        assert "by agent:" not in out

    def test_to_dict_includes_by_agent(self):
        with attribute("planner"):
            record_usage("groq", "llama-3.3-70b-versatile", 10, 5)
        snap = to_dict()
        assert "by_agent" in snap
        assert "planner" in snap["by_agent"]


class TestThreadSafety:
    def test_concurrent_records(self):
        def worker():
            for _ in range(50):
                record_usage("groq", "llama-3.3-70b-versatile", 10, 5)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        totals = get_tracker().total()
        assert totals.calls == 400
        assert totals.prompt_tokens == 400 * 10


class TestIsolatedInstance:
    def test_local_tracker_separate_from_global(self):
        local = CostTracker()
        record_usage("openai", "gpt-4o", 100, 50)  # goes to global
        assert local.total().calls == 0
        assert get_tracker().total().calls == 1
