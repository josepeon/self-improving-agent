"""Tests for InteractionLog schema."""

import json

from self_improving_agent.logging.interaction_log import InteractionLog


class TestInteractionLogDefaults:
    """Tests for default field values."""

    def test_has_interaction_id(self):
        log = InteractionLog()
        assert log.interaction_id
        assert len(log.interaction_id) == 36  # UUID format

    def test_has_timestamp(self):
        log = InteractionLog()
        assert log.timestamp
        assert "T" in log.timestamp  # ISO format

    def test_unique_ids(self):
        log1 = InteractionLog()
        log2 = InteractionLog()
        assert log1.interaction_id != log2.interaction_id

    def test_default_strings_empty(self):
        log = InteractionLog()
        assert log.user_id == ""
        assert log.session_id == ""
        assert log.user_message == ""
        assert log.provider == ""

    def test_default_lists_empty(self):
        log = InteractionLog()
        assert log.conversation_history == []
        assert log.tool_calls == []
        assert log.retrieved_chunks == []

    def test_default_numerics(self):
        log = InteractionLog()
        assert log.temperature == 0.7
        assert log.max_tokens == 150
        assert log.latency_ms == 0.0
        assert log.cost_usd == 0.0

    def test_default_optionals_none(self):
        log = InteractionLog()
        assert log.user_feedback is None
        assert log.failure_category is None
        assert log.quality_score is None
        assert log.time_to_first_token_ms is None


class TestInteractionLogCustom:
    """Tests for creating logs with custom values."""

    def test_custom_fields(self):
        log = InteractionLog(
            user_id="user-123",
            session_id="sess-456",
            user_message="Hello!",
            provider="groq",
            model_name="llama-3.3-70b",
            response_text="Hi there!",
            latency_ms=42.5,
        )
        assert log.user_id == "user-123"
        assert log.provider == "groq"
        assert log.response_text == "Hi there!"
        assert log.latency_ms == 42.5

    def test_feedback_fields(self):
        log = InteractionLog(
            user_feedback="positive",
            user_feedback_text="Great answer!",
            quality_score=0.95,
        )
        assert log.user_feedback == "positive"
        assert log.user_feedback_text == "Great answer!"
        assert log.quality_score == 0.95

    def test_failure_category(self):
        log = InteractionLog(failure_category="hallucination")
        assert log.failure_category == "hallucination"


class TestInteractionLogSerialization:
    """Tests for JSON serialization/deserialization."""

    def test_to_dict(self):
        log = InteractionLog(user_message="Hi", provider="mlx")
        d = log.to_dict()
        assert isinstance(d, dict)
        assert d["user_message"] == "Hi"
        assert d["provider"] == "mlx"
        assert "interaction_id" in d
        assert "timestamp" in d

    def test_to_json_is_valid_json(self):
        log = InteractionLog(user_message="Hello")
        json_str = log.to_json()
        parsed = json.loads(json_str)
        assert parsed["user_message"] == "Hello"

    def test_to_json_single_line(self):
        log = InteractionLog(user_message="Hello")
        json_str = log.to_json()
        assert "\n" not in json_str

    def test_from_json_roundtrip(self):
        original = InteractionLog(
            user_id="user-1",
            user_message="What is AI?",
            provider="groq",
            model_name="llama-70b",
            response_text="AI is...",
            latency_ms=123.4,
            quality_score=0.9,
        )
        json_str = original.to_json()
        restored = InteractionLog.from_json(json_str)

        assert restored.user_id == original.user_id
        assert restored.user_message == original.user_message
        assert restored.provider == original.provider
        assert restored.response_text == original.response_text
        assert restored.latency_ms == original.latency_ms
        assert restored.quality_score == original.quality_score

    def test_from_dict(self):
        data = {
            "user_message": "Hi",
            "provider": "mlx",
            "latency_ms": 50.0,
        }
        log = InteractionLog.from_dict(data)
        assert log.user_message == "Hi"
        assert log.provider == "mlx"
        assert log.latency_ms == 50.0
        # Other fields should have defaults
        assert log.interaction_id  # auto-generated
        assert log.temperature == 0.7

    def test_from_dict_ignores_unknown_keys(self):
        data = {
            "user_message": "Hi",
            "unknown_field": "should be ignored",
        }
        log = InteractionLog.from_dict(data)
        assert log.user_message == "Hi"
        assert not hasattr(log, "unknown_field") or "unknown_field" not in log.to_dict()

    def test_from_json_with_all_fields(self):
        log = InteractionLog(
            user_id="u1",
            session_id="s1",
            request_id="r1",
            user_message="Hello",
            system_prompt="You are helpful.",
            system_prompt_version="v2",
            conversation_history=[{"role": "user", "content": "prev"}],
            memory_context={"key": "value"},
            retrieved_chunks=[{"text": "chunk1", "score": 0.9}],
            provider="groq",
            model_name="llama-70b",
            model_version="v1.0",
            is_fallback=True,
            quality_mode="quality",
            temperature=0.3,
            max_tokens=500,
            tool_calls=[{"name": "search"}],
            tool_outputs=[{"result": "found"}],
            response_text="Answer",
            response_tokens=25,
            latency_ms=200.0,
            time_to_first_token_ms=50.0,
            input_tokens=30,
            cost_usd=0.001,
            user_feedback="positive",
            user_feedback_text="Great!",
            failure_category=None,
            quality_score=0.95,
        )
        restored = InteractionLog.from_json(log.to_json())
        assert restored.user_id == "u1"
        assert restored.is_fallback is True
        assert restored.quality_mode == "quality"
        assert restored.conversation_history == [{"role": "user", "content": "prev"}]
        assert restored.time_to_first_token_ms == 50.0
        assert restored.quality_score == 0.95


class TestInteractionLogFromProviderResponse:
    """Tests for the from_provider_response factory."""

    def test_from_provider_response(self):
        # Simulate a ProviderResponse-like object
        class FakeResponse:
            text = "Hello!"
            provider = "mlx"
            model = "llama-8b"
            is_fallback = False
            tokens_used = 10
            latency_ms = 45.0

        log = InteractionLog.from_provider_response(
            FakeResponse(),
            user_message="Hi",
            system_prompt="Be helpful.",
            quality_mode="fast",
        )
        assert log.user_message == "Hi"
        assert log.system_prompt == "Be helpful."
        assert log.quality_mode == "fast"
        assert log.provider == "mlx"
        assert log.model_name == "llama-8b"
        assert log.response_text == "Hello!"
        assert log.response_tokens == 10
        assert log.latency_ms == 45.0
        assert log.is_fallback is False

    def test_from_provider_response_with_extra_kwargs(self):
        class FakeResponse:
            text = "Answer"
            provider = "groq"
            model = "llama-70b"
            is_fallback = True
            tokens_used = 50
            latency_ms = 200.0

        log = InteractionLog.from_provider_response(
            FakeResponse(),
            user_message="Question",
            user_id="user-42",
            session_id="sess-99",
        )
        assert log.user_id == "user-42"
        assert log.session_id == "sess-99"
        assert log.is_fallback is True
