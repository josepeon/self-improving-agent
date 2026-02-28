"""Tests for prepare_real_data.py — log conversion and session grouping."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from prepare_real_data import (
    conversation_to_alpaca,
    group_into_sessions,
    is_real_interaction,
    session_to_conversation,
)


# ── is_real_interaction ──────────────────────────────────────────────


class TestIsRealInteraction:
    """Tests for filtering genuine user interactions."""

    def test_valid_interaction(self) -> None:
        entry = {
            "user_message": "What is OH?",
            "assistant_response": "OH is a spatial commerce company that builds infrastructure for virtual worlds.",
        }
        assert is_real_interaction(entry) is True

    def test_filters_fact_extraction(self) -> None:
        entry = {
            "user_message": "Analyze this conversation exchange and extract any factual information...",
            "assistant_response": "[]",
        }
        assert is_real_interaction(entry) is False

    def test_filters_empty_user_message(self) -> None:
        entry = {
            "user_message": "",
            "assistant_response": "Hello!",
        }
        assert is_real_interaction(entry) is False

    def test_filters_empty_response(self) -> None:
        entry = {
            "user_message": "Hello there!",
            "assistant_response": "",
        }
        assert is_real_interaction(entry) is False

    def test_filters_very_short_user_message(self) -> None:
        entry = {
            "user_message": "Hi",  # < 5 chars
            "assistant_response": "Hello! How can I help you today?",
        }
        assert is_real_interaction(entry) is False

    def test_filters_very_short_response(self) -> None:
        entry = {
            "user_message": "What is OH?",
            "assistant_response": "A company",  # < 10 chars but == 9... actually this is 9
        }
        assert is_real_interaction(entry) is False

    def test_filters_missing_fields(self) -> None:
        assert is_real_interaction({}) is False
        assert is_real_interaction({"user_message": "Hello"}) is False
        assert is_real_interaction({"assistant_response": "Hi"}) is False

    def test_filters_extract_prefix(self) -> None:
        entry = {
            "user_message": "Extract topics from this conversation",
            "assistant_response": "[]",
        }
        assert is_real_interaction(entry) is False

    def test_filters_summarize_prefix(self) -> None:
        entry = {
            "user_message": "Summarize this conversation for the user",
            "assistant_response": "Summary here",
        }
        assert is_real_interaction(entry) is False


# ── group_into_sessions ──────────────────────────────────────────────


class TestGroupIntoSessions:
    """Tests for grouping log entries into conversation sessions."""

    def test_single_message_session(self) -> None:
        entries = [
            {
                "user_message": "Hello",
                "assistant_response": "Hi!",
                "history_length": 0,
                "timestamp": "2026-02-28T14:00:00+00:00",
            }
        ]
        sessions = group_into_sessions(entries)
        assert len(sessions) == 1
        assert len(sessions[0]) == 1

    def test_multi_turn_session(self) -> None:
        entries = [
            {
                "user_message": "Hello",
                "assistant_response": "Hi!",
                "history_length": 0,
                "timestamp": "2026-02-28T14:00:00+00:00",
            },
            {
                "user_message": "What is OH?",
                "assistant_response": "OH is a company.",
                "history_length": 2,
                "timestamp": "2026-02-28T14:00:30+00:00",
            },
            {
                "user_message": "Tell me more",
                "assistant_response": "Sure, OH does spatial commerce.",
                "history_length": 4,
                "timestamp": "2026-02-28T14:01:00+00:00",
            },
        ]
        sessions = group_into_sessions(entries)
        assert len(sessions) == 1
        assert len(sessions[0]) == 3

    def test_separate_sessions_by_history_length(self) -> None:
        entries = [
            {
                "user_message": "Hello",
                "assistant_response": "Hi!",
                "history_length": 0,
                "timestamp": "2026-02-28T14:00:00+00:00",
            },
            {
                "user_message": "Bye",
                "assistant_response": "Goodbye!",
                "history_length": 0,  # New session
                "timestamp": "2026-02-28T14:00:30+00:00",
            },
        ]
        sessions = group_into_sessions(entries)
        assert len(sessions) == 2

    def test_separate_sessions_by_time_gap(self) -> None:
        entries = [
            {
                "user_message": "Hello",
                "assistant_response": "Hi!",
                "history_length": 0,
                "timestamp": "2026-02-28T14:00:00+00:00",
            },
            {
                "user_message": "Still here?",
                "assistant_response": "Yes!",
                "history_length": 2,
                "timestamp": "2026-02-28T15:00:00+00:00",  # 1 hour gap
            },
        ]
        sessions = group_into_sessions(entries)
        assert len(sessions) == 2

    def test_empty_input(self) -> None:
        assert group_into_sessions([]) == []


# ── session_to_conversation ──────────────────────────────────────────


class TestSessionToConversation:
    """Tests for converting sessions to chat message format."""

    def test_single_turn(self) -> None:
        session = [
            {
                "user_message": "Hello",
                "assistant_response": "Hi there!",
            }
        ]
        conv = session_to_conversation(session)
        assert len(conv) == 2
        assert conv[0] == {"role": "user", "content": "Hello"}
        assert conv[1] == {"role": "assistant", "content": "Hi there!"}

    def test_multi_turn(self) -> None:
        session = [
            {"user_message": "Hello", "assistant_response": "Hi!"},
            {"user_message": "What is OH?", "assistant_response": "A company."},
        ]
        conv = session_to_conversation(session)
        assert len(conv) == 4
        assert conv[0]["role"] == "user"
        assert conv[1]["role"] == "assistant"
        assert conv[2]["role"] == "user"
        assert conv[3]["role"] == "assistant"


# ── conversation_to_alpaca ───────────────────────────────────────────


class TestConversationToAlpaca:
    """Tests for Alpaca-style format conversion."""

    def test_single_turn_no_history(self) -> None:
        conv = [
            {"role": "user", "content": "What is OH?"},
            {"role": "assistant", "content": "OH is a spatial commerce company."},
        ]
        result = conversation_to_alpaca(conv, "You are O-OH.")
        assert result["instruction"] == "You are O-OH."
        assert result["input"] == "What is OH?"
        assert result["output"] == "OH is a spatial commerce company."
        assert result["history"] == []

    def test_multi_turn_with_history(self) -> None:
        conv = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "What is OH?"},
            {"role": "assistant", "content": "OH is a company."},
        ]
        result = conversation_to_alpaca(conv, "system prompt")
        assert result["input"] == "What is OH?"
        assert result["output"] == "OH is a company."
        assert len(result["history"]) == 2
        assert result["history"][0]["content"] == "Hello"
        assert result["history"][1]["content"] == "Hi!"

    def test_with_metadata(self) -> None:
        conv = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        metadata = {"source": "real", "score": 9}
        result = conversation_to_alpaca(conv, "prompt", metadata=metadata)
        assert result["metadata"] == metadata

    def test_without_metadata(self) -> None:
        conv = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = conversation_to_alpaca(conv, "prompt")
        assert "metadata" not in result
