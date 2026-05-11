"""T2.2: feedback store + serving API integration."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from self_improving_agent.feedback.store import (
    Feedback,
    FeedbackRating,
    FeedbackStore,
    bad_interaction_ids,
    feedback_to_sft_pair,
)
from self_improving_agent.providers.base import ProviderResponse
from self_improving_agent.serving import server as server_module
from self_improving_agent.training.registry import (
    ModelRegistry,
    ModelStatus,
    ModelVersion,
)


# ---------------------------------------------------------------------
# Fakes (shared with test_serving.py shape)
# ---------------------------------------------------------------------


class FakeProvider:
    name = "fake"

    async def generate(self, messages, config):
        return ProviderResponse(
            text="ok", provider="fake", model="m",
            latency_ms=0.0, tokens_used=1,
        )

    async def generate_stream(self, *_a, **_kw):  # pragma: no cover
        if False:
            yield ""

    async def health_check(self) -> bool:  # pragma: no cover
        return True


@pytest.fixture
def server_app(tmp_path, monkeypatch):
    """Server with FakeProvider and a populated registry."""
    registry_path = tmp_path / "registry.json"
    registry = ModelRegistry(registry_path)
    registry.register(ModelVersion(
        version="v1.0-test",
        base_model="b",
        status=ModelStatus.PRODUCTION,
    ))
    monkeypatch.setattr(server_module, "_build_provider", lambda v: FakeProvider())
    monkeypatch.setenv("SIA_REGISTRY_PATH", str(registry_path))
    monkeypatch.setenv("SIA_FEEDBACK_PATH", str(tmp_path / "feedback.jsonl"))
    return server_module._make_app()


# ---------------------------------------------------------------------
# Feedback row
# ---------------------------------------------------------------------


class TestFeedbackRow:
    def test_id_is_deterministic(self):
        fb1 = Feedback(
            interaction_id="i1",
            rating=FeedbackRating.BAD,
            user_message="u",
            original_response="r",
            correction="c",
        )
        fb2 = Feedback(
            interaction_id="i1",
            rating=FeedbackRating.BAD,
            user_message="u",
            original_response="r",
            correction="c",
        )
        assert fb1.id == fb2.id

    def test_id_changes_with_rating(self):
        kwargs = {
            "interaction_id": "i1",
            "user_message": "u",
            "original_response": "r",
        }
        good = Feedback(rating=FeedbackRating.GOOD, **kwargs)
        bad = Feedback(rating=FeedbackRating.BAD, **kwargs)
        assert good.id != bad.id

    def test_round_trip(self):
        fb = Feedback(
            interaction_id="i1",
            rating=FeedbackRating.GOOD,
            user_message="hello",
            original_response="hi back",
            metadata={"channel": "slack"},
        )
        restored = Feedback.from_dict(fb.to_dict())
        assert restored.id == fb.id
        assert restored.metadata == {"channel": "slack"}


# ---------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------


class TestFeedbackStore:
    def test_add_and_iter(self, tmp_path):
        store = FeedbackStore(tmp_path / "fb.jsonl")
        fb = Feedback(
            interaction_id="i1",
            rating=FeedbackRating.BAD,
            user_message="u",
            original_response="r",
        )
        assert store.add(fb) is True
        rows = list(store.iter_all())
        assert len(rows) == 1
        assert rows[0].id == fb.id

    def test_duplicate_add_skipped(self, tmp_path):
        store = FeedbackStore(tmp_path / "fb.jsonl")
        fb = Feedback(
            interaction_id="i1",
            rating=FeedbackRating.GOOD,
            user_message="u",
            original_response="r",
        )
        assert store.add(fb) is True
        assert store.add(fb) is False
        assert len(list(store.iter_all())) == 1

    def test_corrupt_line_skipped(self, tmp_path):
        path = tmp_path / "fb.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "interaction_id": "i1",
                "rating": "good",
                "user_message": "u",
                "original_response": "r",
            }) + "\n"
            + "not json at all\n"
        )
        store = FeedbackStore(path)
        rows = list(store.iter_all())
        assert len(rows) == 1

    def test_counts_by_rating(self, tmp_path):
        store = FeedbackStore(tmp_path / "fb.jsonl")
        for i in range(3):
            store.add(Feedback(
                interaction_id=f"i{i}",
                rating=FeedbackRating.GOOD,
                user_message="u",
                original_response="r",
            ))
        store.add(Feedback(
            interaction_id="bad1",
            rating=FeedbackRating.BAD,
            user_message="u",
            original_response="r",
        ))
        counts = store.counts_by_rating()
        assert counts == {"good": 3, "bad": 1}

    def test_counts_by_version(self, tmp_path):
        store = FeedbackStore(tmp_path / "fb.jsonl")
        store.add(Feedback(
            interaction_id="a", rating=FeedbackRating.GOOD,
            user_message="u", original_response="r", model_version="v1.0",
        ))
        store.add(Feedback(
            interaction_id="b", rating=FeedbackRating.BAD,
            user_message="u", original_response="r", model_version="v1.0",
        ))
        store.add(Feedback(
            interaction_id="c", rating=FeedbackRating.GOOD,
            user_message="u", original_response="r", model_version="v1.1",
        ))
        by_version = store.counts_by_version()
        assert by_version["v1.0"] == {"good": 1, "bad": 1}
        assert by_version["v1.1"] == {"good": 1, "bad": 0}


# ---------------------------------------------------------------------
# Dataset builder integration
# ---------------------------------------------------------------------


class TestDatasetBuilderIntegration:
    def test_correction_to_sft_pair(self):
        fb = Feedback(
            interaction_id="i1",
            rating=FeedbackRating.BAD,
            user_message="What's 2+2?",
            original_response="5",
            correction="4",
        )
        pair = feedback_to_sft_pair(fb)
        assert pair is not None
        assert pair["messages"][0]["content"] == "What's 2+2?"
        assert pair["messages"][1]["content"] == "4"
        assert pair["metadata"]["source"] == "feedback"
        assert pair["metadata"]["rating"] == "bad"

    def test_no_correction_returns_none(self):
        fb = Feedback(
            interaction_id="i1",
            rating=FeedbackRating.BAD,
            user_message="u",
            original_response="r",
        )
        assert feedback_to_sft_pair(fb) is None

    def test_empty_correction_returns_none(self):
        fb = Feedback(
            interaction_id="i1",
            rating=FeedbackRating.BAD,
            user_message="u",
            original_response="r",
            correction="   ",
        )
        assert feedback_to_sft_pair(fb) is None

    def test_bad_interaction_ids(self, tmp_path):
        store = FeedbackStore(tmp_path / "fb.jsonl")
        store.add(Feedback(
            interaction_id="bad1", rating=FeedbackRating.BAD,
            user_message="u", original_response="r",
        ))
        store.add(Feedback(
            interaction_id="good1", rating=FeedbackRating.GOOD,
            user_message="u", original_response="r",
        ))
        ids = bad_interaction_ids(store)
        assert ids == {"bad1"}


# ---------------------------------------------------------------------
# Serving API integration
# ---------------------------------------------------------------------


class TestServingFeedbackEndpoint:
    def test_submit_feedback_creates_row(self, server_app):
        with TestClient(server_app) as client:
            response = client.post(
                "/v1/feedback",
                json={
                    "interaction_id": "chatcmpl-abc",
                    "rating": "good",
                    "user_message": "Hi",
                    "original_response": "Hello",
                    "model_version": "v1.0-test",
                },
            )
        assert response.status_code == 200
        body = response.json()
        assert body["accepted"] is True
        assert body["id"].startswith("fb_")

    def test_duplicate_submission_not_accepted(self, server_app):
        payload = {
            "interaction_id": "i1",
            "rating": "bad",
            "user_message": "Hi",
            "original_response": "Bye",
        }
        with TestClient(server_app) as client:
            r1 = client.post("/v1/feedback", json=payload)
            r2 = client.post("/v1/feedback", json=payload)
        assert r1.json()["accepted"] is True
        assert r2.json()["accepted"] is False
        assert r1.json()["id"] == r2.json()["id"]

    def test_stats_aggregates(self, server_app):
        with TestClient(server_app) as client:
            for idx, rating in enumerate(["good", "good", "bad"]):
                client.post(
                    "/v1/feedback",
                    json={
                        "interaction_id": f"i-{idx}",
                        "rating": rating,
                        "user_message": "u",
                        "original_response": "r",
                        "model_version": "v1.0-test",
                    },
                )
            stats = client.get("/v1/feedback/stats").json()
        assert stats["totals"]["good"] == 2
        assert stats["totals"]["bad"] == 1
        assert stats["by_version"]["v1.0-test"]["good"] == 2

    def test_validation_error(self, server_app):
        with TestClient(server_app) as client:
            r = client.post(
                "/v1/feedback",
                json={"interaction_id": "x"},  # missing required fields
            )
        assert r.status_code == 422
