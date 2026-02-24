"""Tests for the @log_llm_interaction middleware decorator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from self_improving_agent.logging.middleware import log_llm_interaction
from self_improving_agent.logging.storage import InteractionLogStore


@pytest.fixture()
def store(tmp_path: Path) -> InteractionLogStore:
    return InteractionLogStore(base_path=str(tmp_path))


def _read_logged(store: InteractionLogStore) -> dict:
    """Read the single logged JSONL entry back as a dict."""
    path = next(iter(store.interactions_path.rglob("*.jsonl")))
    return json.loads(path.read_text().strip().split("\n")[-1])


@dataclass
class FakeResponse:
    """Minimal stand-in for ProviderResponse."""

    text: str = "Hello!"
    provider: str = "groq"
    model: str = "llama-3.3-70b"
    is_fallback: bool = False
    tokens_used: int = 42


# ── Basic decorator behaviour ────────────────────────────────────────


class TestBasicLogging:
    async def test_logs_interaction(self, store: InteractionLogStore) -> None:
        @log_llm_interaction(store)
        async def generate(message: str) -> FakeResponse:
            return FakeResponse()

        await generate("hi")
        data = _read_logged(store)
        assert data["user_message"] == "hi"

    async def test_returns_original_result(self, store: InteractionLogStore) -> None:
        @log_llm_interaction(store)
        async def generate(message: str) -> FakeResponse:
            return FakeResponse(text="world")

        result = await generate("hello")
        assert result.text == "world"

    async def test_captures_provider_fields(self, store: InteractionLogStore) -> None:
        @log_llm_interaction(store)
        async def generate(message: str) -> FakeResponse:
            return FakeResponse(
                provider="mlx", model="llama-3.1-8b", is_fallback=True, tokens_used=99
            )

        await generate("test")
        data = _read_logged(store)
        assert data["provider"] == "mlx"
        assert data["model_name"] == "llama-3.1-8b"
        assert data["is_fallback"] is True
        assert data["response_tokens"] == 99

    async def test_captures_latency(self, store: InteractionLogStore) -> None:
        @log_llm_interaction(store)
        async def generate(message: str) -> FakeResponse:
            return FakeResponse()

        await generate("speed test")
        data = _read_logged(store)
        assert data["latency_ms"] > 0


# ── Kwargs extraction ─────────────────────────────────────────────────


class TestKwargsExtraction:
    async def test_extracts_message_from_kwargs(
        self, store: InteractionLogStore
    ) -> None:
        @log_llm_interaction(store)
        async def generate(*, message: str) -> FakeResponse:
            return FakeResponse()

        await generate(message="from kwargs")
        data = _read_logged(store)
        assert data["user_message"] == "from kwargs"

    async def test_extracts_prompt_kwarg(self, store: InteractionLogStore) -> None:
        @log_llm_interaction(store)
        async def generate(*, prompt: str) -> FakeResponse:
            return FakeResponse()

        await generate(prompt="from prompt kwarg")
        data = _read_logged(store)
        assert data["user_message"] == "from prompt kwarg"

    async def test_extracts_quality_mode(self, store: InteractionLogStore) -> None:
        @log_llm_interaction(store)
        async def generate(message: str, *, quality_mode: str = "fast") -> FakeResponse:
            return FakeResponse()

        await generate("x", quality_mode="quality")
        data = _read_logged(store)
        assert data["quality_mode"] == "quality"

    async def test_extracts_system_prompt(self, store: InteractionLogStore) -> None:
        @log_llm_interaction(store)
        async def generate(message: str, *, system_prompt: str = "") -> FakeResponse:
            return FakeResponse()

        await generate("x", system_prompt="You are helpful")
        data = _read_logged(store)
        assert data["system_prompt"] == "You are helpful"

    async def test_extracts_request_id(self, store: InteractionLogStore) -> None:
        @log_llm_interaction(store)
        async def generate(message: str, *, request_id: str = "") -> FakeResponse:
            return FakeResponse()

        await generate("x", request_id="req-123")
        data = _read_logged(store)
        assert data["request_id"] == "req-123"

    async def test_no_args_empty_message(self, store: InteractionLogStore) -> None:
        @log_llm_interaction(store)
        async def generate() -> FakeResponse:
            return FakeResponse()

        await generate()
        data = _read_logged(store)
        assert data["user_message"] == ""


# ── Error handling ────────────────────────────────────────────────────


class TestErrorHandling:
    async def test_logs_on_exception(self, store: InteractionLogStore) -> None:
        @log_llm_interaction(store)
        async def generate(message: str) -> FakeResponse:
            msg = "boom"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="boom"):
            await generate("error test")

        data = _read_logged(store)
        assert data["failure_category"] == "error"
        assert "RuntimeError: boom" in data["response_text"]
        assert data["latency_ms"] > 0

    async def test_exception_propagates(self, store: InteractionLogStore) -> None:
        @log_llm_interaction(store)
        async def generate(message: str) -> FakeResponse:
            msg = "propagated"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="propagated"):
            await generate("test")


# ── Custom extractors ────────────────────────────────────────────────


class TestCustomExtractors:
    async def test_custom_message_extractor(self, store: InteractionLogStore) -> None:
        def extract(args: tuple, kwargs: dict) -> str:
            return f"custom: {args[0]}"

        @log_llm_interaction(store, extract_message=extract)
        async def generate(msg: str) -> FakeResponse:
            return FakeResponse()

        await generate("raw")
        data = _read_logged(store)
        assert data["user_message"] == "custom: raw"

    async def test_custom_system_prompt_extractor(
        self, store: InteractionLogStore
    ) -> None:
        def extract(args: tuple, kwargs: dict) -> str:
            return "custom system prompt"

        @log_llm_interaction(store, extract_system_prompt=extract)
        async def generate(msg: str) -> FakeResponse:
            return FakeResponse()

        await generate("x")
        data = _read_logged(store)
        assert data["system_prompt"] == "custom system prompt"


# ── Preserves function metadata ──────────────────────────────────────


class TestFunctionMetadata:
    async def test_preserves_name(self, store: InteractionLogStore) -> None:
        @log_llm_interaction(store)
        async def my_generate(message: str) -> FakeResponse:
            """My docstring."""
            return FakeResponse()

        assert my_generate.__name__ == "my_generate"
        assert my_generate.__doc__ == "My docstring."
