"""Tests for DatasetBuilder."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from self_improving_agent.logging.interaction_log import InteractionLog
from self_improving_agent.training.dataset_builder import DatasetBuilder


def _make_log(**overrides: object) -> InteractionLog:
    """Create an InteractionLog with sensible defaults and optional overrides."""
    defaults = {
        "user_message": "What is 2+2?",
        "system_prompt": "You are a helpful assistant.",
        "response_text": "4",
        "user_feedback": "positive",
        "quality_score": 0.9,
    }
    defaults.update(overrides)
    return InteractionLog(**defaults)  # type: ignore[arg-type]


def _mock_store(logs: list[InteractionLog]) -> AsyncMock:
    store = AsyncMock()
    store.get_interactions = AsyncMock(return_value=logs)
    return store


NOW = datetime.now(UTC)
YESTERDAY = datetime(2026, 2, 23, tzinfo=UTC)


# ── SFT dataset ──────────────────────────────────────────────────────


class TestBuildSFTDataset:
    async def test_positive_feedback_included(self) -> None:
        log = _make_log(user_feedback="positive", quality_score=None)
        builder = DatasetBuilder(_mock_store([log]))
        ds = await builder.build_sft_dataset(YESTERDAY, NOW)
        assert len(ds) == 1
        assert ds[0]["output"] == "4"

    async def test_high_quality_score_included(self) -> None:
        log = _make_log(user_feedback=None, quality_score=0.95)
        builder = DatasetBuilder(_mock_store([log]))
        ds = await builder.build_sft_dataset(YESTERDAY, NOW)
        assert len(ds) == 1

    async def test_negative_feedback_excluded(self) -> None:
        log = _make_log(user_feedback="negative", quality_score=0.95)
        builder = DatasetBuilder(_mock_store([log]))
        ds = await builder.build_sft_dataset(YESTERDAY, NOW)
        assert len(ds) == 0

    async def test_failure_category_excluded(self) -> None:
        log = _make_log(failure_category="hallucination")
        builder = DatasetBuilder(_mock_store([log]))
        ds = await builder.build_sft_dataset(YESTERDAY, NOW)
        assert len(ds) == 0

    async def test_empty_response_excluded(self) -> None:
        log = _make_log(response_text="")
        builder = DatasetBuilder(_mock_store([log]))
        ds = await builder.build_sft_dataset(YESTERDAY, NOW)
        assert len(ds) == 0

    async def test_no_signal_excluded(self) -> None:
        """No positive feedback and no quality score -> excluded."""
        log = _make_log(user_feedback=None, quality_score=None)
        builder = DatasetBuilder(_mock_store([log]))
        ds = await builder.build_sft_dataset(YESTERDAY, NOW)
        assert len(ds) == 0

    async def test_custom_min_quality(self) -> None:
        log = _make_log(user_feedback=None, quality_score=0.7)
        builder = DatasetBuilder(_mock_store([log]))
        # 0.7 < default 0.8 -> excluded
        assert len(await builder.build_sft_dataset(YESTERDAY, NOW)) == 0
        # lower threshold -> included
        assert (
            len(await builder.build_sft_dataset(YESTERDAY, NOW, min_quality_score=0.6))
            == 1
        )

    async def test_alpaca_format(self) -> None:
        log = _make_log(
            system_prompt="Be concise.",
            user_message="Hi",
            response_text="Hello!",
            conversation_history=[{"role": "user", "content": "Hey"}],
        )
        builder = DatasetBuilder(_mock_store([log]))
        ds = await builder.build_sft_dataset(YESTERDAY, NOW)
        item = ds[0]
        assert item["instruction"] == "Be concise."
        assert item["input"] == "Hi"
        assert item["output"] == "Hello!"
        assert len(item["history"]) == 1

    async def test_empty_store(self) -> None:
        builder = DatasetBuilder(_mock_store([]))
        ds = await builder.build_sft_dataset(YESTERDAY, NOW)
        assert ds == []


# ── SFT save to file ─────────────────────────────────────────────────


class TestBuildSFTAndSave:
    async def test_writes_jsonl(self, tmp_path: object) -> None:
        import json
        from pathlib import Path

        out = Path(str(tmp_path)) / "train.jsonl"
        log = _make_log()
        builder = DatasetBuilder(_mock_store([log]))
        count = await builder.build_sft_and_save(YESTERDAY, NOW, out)
        assert count == 1
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["output"] == "4"


# ── DPO dataset ──────────────────────────────────────────────────────


class TestBuildDPODataset:
    async def test_pairs_matched_by_message(self) -> None:
        pos = _make_log(user_feedback="positive", response_text="Good answer")
        neg = _make_log(
            user_feedback="negative",
            response_text="Bad answer",
            quality_score=None,
        )
        builder = DatasetBuilder(_mock_store([pos, neg]))
        pairs = await builder.build_dpo_dataset(YESTERDAY, NOW)
        assert len(pairs) == 1
        assert pairs[0]["chosen"] == "Good answer"
        assert pairs[0]["rejected"] == "Bad answer"

    async def test_no_pairs_without_match(self) -> None:
        pos = _make_log(user_message="Question A", user_feedback="positive")
        neg = _make_log(user_message="Question B", user_feedback="negative")
        builder = DatasetBuilder(_mock_store([pos, neg]))
        pairs = await builder.build_dpo_dataset(YESTERDAY, NOW)
        assert len(pairs) == 0

    async def test_empty_response_excluded(self) -> None:
        pos = _make_log(user_feedback="positive", response_text="")
        neg = _make_log(user_feedback="negative", response_text="bad")
        builder = DatasetBuilder(_mock_store([pos, neg]))
        pairs = await builder.build_dpo_dataset(YESTERDAY, NOW)
        assert len(pairs) == 0


# ── Stats ─────────────────────────────────────────────────────────────


class TestDatasetStats:
    async def test_stats_counts(self) -> None:
        logs = [
            _make_log(user_feedback="positive"),
            _make_log(user_feedback="negative", failure_category="hallucination"),
            _make_log(user_feedback=None, quality_score=None),
        ]
        builder = DatasetBuilder(_mock_store(logs))
        stats = await builder.dataset_stats(YESTERDAY, NOW)
        assert stats["total_interactions"] == 3
        assert stats["sft_eligible"] == 1
        assert stats["with_feedback"] == 2
        assert stats["with_failures"] == 1


# ── Eligibility helper ───────────────────────────────────────────────


class TestSFTEligibility:
    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"user_feedback": "positive", "quality_score": None}, True),
            ({"user_feedback": None, "quality_score": 0.9}, True),
            ({"user_feedback": None, "quality_score": 0.5}, False),
            ({"user_feedback": "negative", "quality_score": 0.95}, False),
            ({"failure_category": "refusal"}, False),
            ({"response_text": ""}, False),
            ({"user_message": ""}, False),
        ],
    )
    def test_eligibility(self, kwargs: dict[str, object], expected: bool) -> None:
        log = _make_log(**kwargs)
        assert DatasetBuilder._is_sft_eligible(log, 0.8) is expected
