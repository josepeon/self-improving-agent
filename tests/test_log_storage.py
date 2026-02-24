"""Tests for InteractionLogStore — async JSONL storage."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from self_improving_agent.logging.interaction_log import InteractionLog
from self_improving_agent.logging.storage import InteractionLogStore


@pytest.fixture()
def store(tmp_path: Path) -> InteractionLogStore:
    """Return a store rooted in a temp directory."""
    return InteractionLogStore(base_path=str(tmp_path))


def _make_log(**overrides: object) -> InteractionLog:
    """Helper to create a log with sensible defaults."""
    defaults = {
        "user_message": "hello",
        "provider": "groq",
        "model_name": "llama-3.3-70b",
        "response_text": "Hi there!",
    }
    defaults.update(overrides)
    return InteractionLog(**defaults)  # type: ignore[arg-type]


# ── log_interaction ───────────────────────────────────────────────────


class TestLogInteraction:
    async def test_creates_jsonl_file(self, store: InteractionLogStore) -> None:
        log = _make_log()
        path = await store.log_interaction(log)
        assert path.exists()
        assert path.name == "interactions.jsonl"

    async def test_file_content_is_valid_json(self, store: InteractionLogStore) -> None:
        log = _make_log()
        path = await store.log_interaction(log)
        content = path.read_text().strip()
        data = json.loads(content)
        assert data["user_message"] == "hello"

    async def test_appends_multiple_logs(self, store: InteractionLogStore) -> None:
        log1 = _make_log(user_message="first")
        log2 = _make_log(user_message="second")
        path = await store.log_interaction(log1)
        await store.log_interaction(log2)
        lines = [x for x in path.read_text().strip().split("\n") if x]
        assert len(lines) == 2
        assert json.loads(lines[0])["user_message"] == "first"
        assert json.loads(lines[1])["user_message"] == "second"

    async def test_date_partitioned_path(self, store: InteractionLogStore) -> None:
        ts = "2026-02-24T10:30:00+00:00"
        log = _make_log(timestamp=ts)
        path = await store.log_interaction(log)
        assert "2026/02/24" in str(path)

    async def test_returns_path(self, store: InteractionLogStore) -> None:
        log = _make_log()
        result = await store.log_interaction(log)
        assert isinstance(result, Path)


# ── get_interactions ──────────────────────────────────────────────────


class TestGetInteractions:
    async def test_returns_all_in_range(self, store: InteractionLogStore) -> None:
        ts = "2026-02-24T12:00:00+00:00"
        for i in range(3):
            await store.log_interaction(
                _make_log(user_message=f"msg-{i}", timestamp=ts)
            )
        results = await store.get_interactions(
            start_date=datetime(2026, 2, 24, tzinfo=UTC),
            end_date=datetime(2026, 2, 24, tzinfo=UTC),
        )
        assert len(results) == 3

    async def test_empty_range(self, store: InteractionLogStore) -> None:
        results = await store.get_interactions(
            start_date=datetime(2026, 1, 1, tzinfo=UTC),
            end_date=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert results == []

    async def test_filter_by_user_id(self, store: InteractionLogStore) -> None:
        ts = "2026-02-24T12:00:00+00:00"
        await store.log_interaction(_make_log(user_id="alice", timestamp=ts))
        await store.log_interaction(_make_log(user_id="bob", timestamp=ts))
        results = await store.get_interactions(
            start_date=datetime(2026, 2, 24, tzinfo=UTC),
            end_date=datetime(2026, 2, 24, tzinfo=UTC),
            user_id="alice",
        )
        assert len(results) == 1
        assert results[0].user_id == "alice"

    async def test_filter_has_feedback_true(self, store: InteractionLogStore) -> None:
        ts = "2026-02-24T12:00:00+00:00"
        await store.log_interaction(_make_log(user_feedback="positive", timestamp=ts))
        await store.log_interaction(_make_log(timestamp=ts))
        results = await store.get_interactions(
            start_date=datetime(2026, 2, 24, tzinfo=UTC),
            end_date=datetime(2026, 2, 24, tzinfo=UTC),
            has_feedback=True,
        )
        assert len(results) == 1
        assert results[0].user_feedback == "positive"

    async def test_filter_has_feedback_false(self, store: InteractionLogStore) -> None:
        ts = "2026-02-24T12:00:00+00:00"
        await store.log_interaction(_make_log(user_feedback="positive", timestamp=ts))
        await store.log_interaction(_make_log(timestamp=ts))
        results = await store.get_interactions(
            start_date=datetime(2026, 2, 24, tzinfo=UTC),
            end_date=datetime(2026, 2, 24, tzinfo=UTC),
            has_feedback=False,
        )
        assert len(results) == 1
        assert results[0].user_feedback is None

    async def test_filter_failure_category(self, store: InteractionLogStore) -> None:
        ts = "2026-02-24T12:00:00+00:00"
        await store.log_interaction(
            _make_log(failure_category="hallucination", timestamp=ts)
        )
        await store.log_interaction(_make_log(timestamp=ts))
        results = await store.get_interactions(
            start_date=datetime(2026, 2, 24, tzinfo=UTC),
            end_date=datetime(2026, 2, 24, tzinfo=UTC),
            failure_category="hallucination",
        )
        assert len(results) == 1
        assert results[0].failure_category == "hallucination"

    async def test_multi_day_range(self, store: InteractionLogStore) -> None:
        for day in (23, 24, 25):
            ts = f"2026-02-{day:02d}T12:00:00+00:00"
            await store.log_interaction(_make_log(timestamp=ts))
        results = await store.get_interactions(
            start_date=datetime(2026, 2, 23, tzinfo=UTC),
            end_date=datetime(2026, 2, 25, tzinfo=UTC),
        )
        assert len(results) == 3


# ── get_interaction_by_id ─────────────────────────────────────────────


class TestGetInteractionById:
    async def test_find_with_date_hint(self, store: InteractionLogStore) -> None:
        ts = "2026-02-24T12:00:00+00:00"
        log = _make_log(timestamp=ts)
        await store.log_interaction(log)
        found = await store.get_interaction_by_id(
            log.interaction_id,
            date_hint=datetime(2026, 2, 24, tzinfo=UTC),
        )
        assert found is not None
        assert found.interaction_id == log.interaction_id

    async def test_find_without_date_hint(self, store: InteractionLogStore) -> None:
        ts = "2026-02-24T12:00:00+00:00"
        log = _make_log(timestamp=ts)
        await store.log_interaction(log)
        found = await store.get_interaction_by_id(log.interaction_id)
        assert found is not None
        assert found.interaction_id == log.interaction_id

    async def test_not_found(self, store: InteractionLogStore) -> None:
        result = await store.get_interaction_by_id("nonexistent-id")
        assert result is None

    async def test_not_found_with_date_hint(self, store: InteractionLogStore) -> None:
        result = await store.get_interaction_by_id(
            "nonexistent-id",
            date_hint=datetime(2026, 2, 24, tzinfo=UTC),
        )
        assert result is None


# ── add_feedback ──────────────────────────────────────────────────────


class TestAddFeedback:
    async def test_creates_pending_file(self, store: InteractionLogStore) -> None:
        await store.add_feedback("id-1", "positive")
        pending = store.feedback_path / "feedback_pending.jsonl"
        assert pending.exists()

    async def test_pending_content(self, store: InteractionLogStore) -> None:
        await store.add_feedback("id-1", "negative", feedback_text="bad answer")
        pending = store.feedback_path / "feedback_pending.jsonl"
        data = json.loads(pending.read_text().strip())
        assert data["interaction_id"] == "id-1"
        assert data["feedback"] == "negative"
        assert data["feedback_text"] == "bad answer"
        assert "timestamp" in data

    async def test_appends_multiple(self, store: InteractionLogStore) -> None:
        await store.add_feedback("id-1", "positive")
        await store.add_feedback("id-2", "negative")
        pending = store.feedback_path / "feedback_pending.jsonl"
        lines = [x for x in pending.read_text().strip().split("\n") if x]
        assert len(lines) == 2


# ── flush_feedback ────────────────────────────────────────────────────


class TestFlushFeedback:
    async def test_no_pending_returns_zero(self, store: InteractionLogStore) -> None:
        assert await store.flush_feedback() == 0

    async def test_merges_feedback_into_log(self, store: InteractionLogStore) -> None:
        ts = "2026-02-24T12:00:00+00:00"
        log = _make_log(timestamp=ts)
        await store.log_interaction(log)
        await store.add_feedback(log.interaction_id, "positive", "great!")

        merged = await store.flush_feedback()
        assert merged == 1

        # Verify the interaction was updated
        found = await store.get_interaction_by_id(
            log.interaction_id,
            date_hint=datetime(2026, 2, 24, tzinfo=UTC),
        )
        assert found is not None
        assert found.user_feedback == "positive"
        assert found.user_feedback_text == "great!"

    async def test_clears_pending_after_full_merge(
        self, store: InteractionLogStore
    ) -> None:
        ts = "2026-02-24T12:00:00+00:00"
        log = _make_log(timestamp=ts)
        await store.log_interaction(log)
        await store.add_feedback(log.interaction_id, "positive")

        await store.flush_feedback()
        pending = store.feedback_path / "feedback_pending.jsonl"
        assert pending.read_text().strip() == ""

    async def test_unmerged_feedback_kept(self, store: InteractionLogStore) -> None:
        # Feedback for a nonexistent interaction stays in pending
        await store.add_feedback("nonexistent-id", "negative")
        merged = await store.flush_feedback()
        assert merged == 0
        pending = store.feedback_path / "feedback_pending.jsonl"
        lines = [x for x in pending.read_text().strip().split("\n") if x]
        assert len(lines) == 1


# ── count_interactions ────────────────────────────────────────────────


class TestCountInteractions:
    async def test_count_matches_logged(self, store: InteractionLogStore) -> None:
        ts = "2026-02-24T12:00:00+00:00"
        for _ in range(5):
            await store.log_interaction(_make_log(timestamp=ts))
        count = await store.count_interactions(
            start_date=datetime(2026, 2, 24, tzinfo=UTC),
            end_date=datetime(2026, 2, 24, tzinfo=UTC),
        )
        assert count == 5

    async def test_empty_range_returns_zero(self, store: InteractionLogStore) -> None:
        count = await store.count_interactions(
            start_date=datetime(2026, 1, 1, tzinfo=UTC),
            end_date=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert count == 0


# ── _matches (static) ────────────────────────────────────────────────


class TestMatches:
    def test_no_filters_matches_all(self) -> None:
        assert InteractionLogStore._matches(_make_log(), None, None, None)

    def test_user_id_match(self) -> None:
        log = _make_log(user_id="alice")
        assert InteractionLogStore._matches(log, "alice", None, None)
        assert not InteractionLogStore._matches(log, "bob", None, None)

    def test_has_feedback_true(self) -> None:
        log_with = _make_log(user_feedback="positive")
        log_without = _make_log()
        assert InteractionLogStore._matches(log_with, None, True, None)
        assert not InteractionLogStore._matches(log_without, None, True, None)

    def test_has_feedback_false(self) -> None:
        log_with = _make_log(user_feedback="positive")
        log_without = _make_log()
        assert not InteractionLogStore._matches(log_with, None, False, None)
        assert InteractionLogStore._matches(log_without, None, False, None)

    def test_failure_category_match(self) -> None:
        log = _make_log(failure_category="hallucination")
        assert InteractionLogStore._matches(log, None, None, "hallucination")
        assert not InteractionLogStore._matches(log, None, None, "refusal")
