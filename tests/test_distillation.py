"""T2.3: distillation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_improving_agent.providers.base import ProviderResponse
from self_improving_agent.training.distillation import (
    DistilledPair,
    DistillationRunner,
    JsonlQuestionPool,
    ListQuestionPool,
    run_distillation,
)
from self_improving_agent.training.judges import EnsembleJudge, ProviderJudge


# ---------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------


class FakeProvider:
    def __init__(self, text: str = "ok response", *, name: str = "fake"):
        self.text = text
        self.name = name
        self.model = "fake-model"
        self.calls = 0

    async def generate(self, messages, config):
        self.calls += 1
        return ProviderResponse(
            text=self.text,
            provider=self.name,
            model=self.model,
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
        raise RuntimeError("teacher unavailable")


# ---------------------------------------------------------------------
# Question pools
# ---------------------------------------------------------------------


class TestQuestionPools:
    def test_list_pool(self):
        pool = ListQuestionPool(prompts=["a", "b", "c"])
        assert list(pool) == ["a", "b", "c"]

    def test_jsonl_pool(self, tmp_path):
        path = tmp_path / "q.jsonl"
        path.write_text(
            json.dumps({"prompt": "what is X?"}) + "\n"
            + json.dumps({"question": "explain Y"}) + "\n"
            + json.dumps({"user": "tell me Z"}) + "\n"
            + json.dumps({"unrelated": "no good field"}) + "\n"
        )
        pool = JsonlQuestionPool(path)
        assert list(pool) == ["what is X?", "explain Y", "tell me Z"]

    def test_jsonl_pool_skips_corrupt_lines(self, tmp_path):
        path = tmp_path / "q.jsonl"
        path.write_text(
            json.dumps({"prompt": "good"}) + "\n"
            + "this is not json\n"
            + json.dumps({"prompt": "also good"}) + "\n"
        )
        assert list(JsonlQuestionPool(path)) == ["good", "also good"]


# ---------------------------------------------------------------------
# DistilledPair / report shapes
# ---------------------------------------------------------------------


class TestDistilledPair:
    def test_sft_row_shape(self):
        pair = DistilledPair(
            prompt="hi",
            response="hello",
            judge_score=8.5,
            teacher_model="m",
            teacher_provider="p",
        )
        row = pair.to_sft_row()
        assert row["messages"][0] == {"role": "user", "content": "hi"}
        assert row["messages"][1] == {"role": "assistant", "content": "hello"}
        assert row["metadata"]["source"] == "distillation"
        assert row["metadata"]["judge_score"] == 8.5


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------


class TestDistillationRunner:
    @pytest.mark.asyncio
    async def test_kept_pair_written(self, tmp_path):
        teacher = FakeProvider(text="great answer that exceeds the bar")
        # Judge returns 9 → above the 7 threshold
        judge = EnsembleJudge(
            judges=[ProviderJudge(FakeProvider(text="9"), name="j")],
            threshold=7.0,
        )
        out = tmp_path / "out.jsonl"
        runner = DistillationRunner(teacher=teacher, judge=judge, output_path=out)

        pool = ListQuestionPool(prompts=["what is X?"])
        report = await runner.run(pool)

        assert report.pairs_kept == 1
        assert report.pairs_dropped == 0
        row = json.loads(out.read_text())
        assert row["messages"][0]["content"] == "what is X?"
        assert row["metadata"]["teacher_provider"] == "fake"

    @pytest.mark.asyncio
    async def test_below_threshold_dropped(self, tmp_path):
        teacher = FakeProvider()
        judge = EnsembleJudge(
            judges=[ProviderJudge(FakeProvider(text="3"), name="strict")],
            threshold=7.0,
        )
        out = tmp_path / "out.jsonl"
        runner = DistillationRunner(teacher=teacher, judge=judge, output_path=out)
        report = await runner.run(ListQuestionPool(prompts=["x"]))

        assert report.pairs_kept == 0
        assert report.pairs_dropped == 1
        assert out.exists()
        assert out.read_text() == ""

    @pytest.mark.asyncio
    async def test_teacher_error_counted(self, tmp_path):
        teacher = FailingProvider()
        judge = EnsembleJudge(
            judges=[ProviderJudge(FakeProvider(text="9"))], threshold=7.0,
        )
        runner = DistillationRunner(
            teacher=teacher, judge=judge, output_path=tmp_path / "out.jsonl"
        )
        report = await runner.run(ListQuestionPool(prompts=["x", "y", "z"]))

        assert report.questions_seen == 3
        assert report.teacher_errors == 3
        assert report.pairs_kept == 0
        assert report.pairs_dropped == 3

    @pytest.mark.asyncio
    async def test_report_includes_judge_calibration(self, tmp_path):
        teacher = FakeProvider(text="good")
        judge = EnsembleJudge(
            judges=[ProviderJudge(FakeProvider(text="8"), name="j1")],
            threshold=7.0,
        )
        runner = DistillationRunner(
            teacher=teacher, judge=judge, output_path=tmp_path / "out.jsonl"
        )
        report = await runner.run(ListQuestionPool(prompts=["a", "b"]))

        assert "j1" in report.judge_calibration
        assert report.judge_calibration["j1"]["count"] == 2

    @pytest.mark.asyncio
    async def test_system_prompt_prepended(self, tmp_path):
        captured: list[list] = []

        class CapturingProvider(FakeProvider):
            async def generate(self, messages, config):
                captured.append(messages)
                return await super().generate(messages, config)

        teacher = CapturingProvider(text="x")
        judge = EnsembleJudge(
            judges=[ProviderJudge(FakeProvider(text="9"))], threshold=7.0,
        )
        runner = DistillationRunner(
            teacher=teacher,
            judge=judge,
            output_path=tmp_path / "out.jsonl",
            system_prompt="You are friendly.",
        )
        await runner.run(ListQuestionPool(prompts=["hi"]))

        assert captured[0][0]["role"] == "system"
        assert captured[0][0]["content"] == "You are friendly."

    @pytest.mark.asyncio
    async def test_report_render_contains_headline(self, tmp_path):
        runner = DistillationRunner(
            teacher=FakeProvider(text="x"),
            judge=EnsembleJudge(
                judges=[ProviderJudge(FakeProvider(text="9"))], threshold=7.0,
            ),
            output_path=tmp_path / "out.jsonl",
        )
        report = await runner.run(ListQuestionPool(prompts=["a"]))
        out = report.render()
        assert "questions seen" in out
        assert "pairs kept" in out


# ---------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------


class TestRunDistillation:
    @pytest.mark.asyncio
    async def test_high_level_helper(self, tmp_path):
        teacher = FakeProvider(text="hello")
        judges = [ProviderJudge(FakeProvider(text="9"), name="j1")]

        report = await run_distillation(
            pool=ListQuestionPool(prompts=["q1", "q2"]),
            output_path=tmp_path / "out.jsonl",
            teacher=teacher,
            judges=judges,
            judge_threshold=7.0,
        )

        assert report.pairs_kept == 2
