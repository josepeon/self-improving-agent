"""Tests for sync_logs.py — filtering, extraction, and real interaction detection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from sync_logs import (
    extract_sia_entries,
    filter_real_interactions,
)


# ── filter_real_interactions ─────────────────────────────────────────


class TestFilterRealInteractions:
    """Tests for filtering out internal system calls."""

    def test_keeps_real_user_message(self) -> None:
        entries = [
            {
                "timestamp": "2026-02-28T14:31:30.707096+00:00",
                "user_message": "Hey, how's it going?",
                "assistant_response": "I'm doing well!",
                "model": "llama-3.3-70b-versatile",
            }
        ]
        result = filter_real_interactions(entries)
        assert len(result) == 1
        assert result[0]["user_message"] == "Hey, how's it going?"

    def test_filters_fact_extraction(self) -> None:
        entries = [
            {
                "timestamp": "2026-02-28T14:31:30.989814+00:00",
                "user_message": "Analyze this conversation exchange and extract any factual information about the user.\nOnly extract facts...",
                "assistant_response": "[]",
                "model": "llama-3.3-70b-versatile",
            }
        ]
        result = filter_real_interactions(entries)
        assert len(result) == 0

    def test_filters_extract_prefix(self) -> None:
        entries = [
            {
                "user_message": "Extract key topics from this conversation",
                "assistant_response": "[]",
            }
        ]
        result = filter_real_interactions(entries)
        assert len(result) == 0

    def test_filters_summarize_prefix(self) -> None:
        entries = [
            {
                "user_message": "Summarize this conversation for the user profile",
                "assistant_response": "Summary...",
            }
        ]
        result = filter_real_interactions(entries)
        assert len(result) == 0

    def test_mixed_real_and_internal(self) -> None:
        entries = [
            {
                "user_message": "Hello, how are you?",
                "assistant_response": "I'm doing well!",
            },
            {
                "user_message": "Analyze this conversation exchange and extract...",
                "assistant_response": "[]",
            },
            {
                "user_message": "What is OH?",
                "assistant_response": "OH is a spatial commerce company.",
            },
            {
                "user_message": "Extract facts from this...",
                "assistant_response": "[]",
            },
        ]
        result = filter_real_interactions(entries)
        assert len(result) == 2
        assert result[0]["user_message"] == "Hello, how are you?"
        assert result[1]["user_message"] == "What is OH?"

    def test_empty_input(self) -> None:
        assert filter_real_interactions([]) == []


# ── extract_sia_entries ──────────────────────────────────────────────


class TestExtractSiaEntries:
    """Tests for parsing SIA log entries from Railway output."""

    def test_railway_json_format(self) -> None:
        entry = {"user_message": "Hello", "assistant_response": "Hi!"}
        railway_line = json.dumps(
            {
                "message": f"SIA_LOG:{json.dumps(entry)}",
                "timestamp": "2026-02-28T00:00:00Z",
            }
        )
        result = extract_sia_entries([railway_line])
        assert len(result) == 1
        assert result[0]["user_message"] == "Hello"

    def test_raw_text_format(self) -> None:
        entry = {"user_message": "Hello", "assistant_response": "Hi!"}
        raw_line = f"SIA_LOG:{json.dumps(entry)}"
        result = extract_sia_entries([raw_line])
        assert len(result) == 1
        assert result[0]["user_message"] == "Hello"

    def test_non_sia_lines_ignored(self) -> None:
        lines = [
            '{"message": "Starting server on port 8000", "timestamp": "2026-02-28"}',
            '{"message": "Health check OK", "timestamp": "2026-02-28"}',
        ]
        result = extract_sia_entries(lines)
        assert len(result) == 0

    def test_empty_lines_handled(self) -> None:
        result = extract_sia_entries(["", "  ", "\n"])
        assert len(result) == 0

    def test_invalid_json_handled(self) -> None:
        result = extract_sia_entries(["not json at all", "{{bad json}}"])
        assert len(result) == 0

    def test_multiple_entries(self) -> None:
        entries = [
            {"user_message": f"msg{i}", "assistant_response": f"resp{i}"}
            for i in range(5)
        ]
        lines = [json.dumps({"message": f"SIA_LOG:{json.dumps(e)}"}) for e in entries]
        result = extract_sia_entries(lines)
        assert len(result) == 5

    def test_mixed_valid_and_invalid(self) -> None:
        valid = {"user_message": "Hello", "assistant_response": "Hi!"}
        lines = [
            "random log line",
            json.dumps({"message": f"SIA_LOG:{json.dumps(valid)}"}),
            '{"message": "not a SIA log"}',
            "SIA_LOG:{bad json here",
        ]
        result = extract_sia_entries(lines)
        assert len(result) == 1
        assert result[0]["user_message"] == "Hello"
