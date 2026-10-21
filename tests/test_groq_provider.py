"""Tests for GroqProvider (mocked — no real API calls)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from self_improving_agent.providers.base import GenerationConfig, ProviderResponse
from self_improving_agent.providers.groq_provider import GroqProvider


@pytest.fixture
def groq_provider():
    """Create a GroqProvider with a fake API key."""
    return GroqProvider(api_key="test-key-123", model="llama-3.3-70b-versatile")


def _mock_completion(text="Hello!", tokens=10, finish_reason="stop"):
    """Create a mock Groq completion response."""
    choice = MagicMock()
    choice.message.content = text
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.total_tokens = tokens
    usage.prompt_tokens = 5
    usage.completion_tokens = tokens - 5

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _mock_stream_chunks(tokens=None):
    """Create mock streaming chunks."""
    if tokens is None:
        tokens = ["Hello", " world", "!"]

    chunks = []
    for token in tokens:
        chunk = MagicMock()
        delta = MagicMock()
        delta.content = token
        chunk.choices = [MagicMock(delta=delta)]
        chunks.append(chunk)

    # Final chunk with no content
    final = MagicMock()
    final_delta = MagicMock()
    final_delta.content = None
    final.choices = [MagicMock(delta=final_delta)]
    chunks.append(final)

    return chunks


class TestGroqProviderInit:
    """Tests for GroqProvider initialization."""

    def test_name(self, groq_provider):
        assert groq_provider.name == "groq"

    def test_default_model(self, monkeypatch):
        monkeypatch.delenv("GROQ_MODEL", raising=False)
        monkeypatch.delenv("GROQ_FALLBACK_MODEL", raising=False)
        provider = GroqProvider(api_key="key")
        assert provider._model == "openai/gpt-oss-120b"
        assert provider._fallback_model == "openai/gpt-oss-20b"

    def test_stale_env_pin_is_ignored(self, monkeypatch):
        """A retired model left in .env must not be used (it can only fail)."""
        monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        provider = GroqProvider(api_key="key")
        assert provider._model == "openai/gpt-oss-120b"

    def test_live_env_pin_is_respected(self, monkeypatch):
        monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
        assert GroqProvider(api_key="key")._model == "openai/gpt-oss-20b"

    def test_custom_model(self):
        provider = GroqProvider(api_key="key", model="custom-model")
        assert provider._model == "custom-model"

    def test_no_api_key_raises_on_use(self):
        provider = GroqProvider(api_key="")
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            provider._get_client()


class TestGroqProviderGenerate:
    """Tests for GroqProvider.generate()."""

    @pytest.mark.asyncio
    async def test_generate_returns_response(self, groq_provider):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("Hello there!", 15)
        )
        groq_provider._client = mock_client

        response = await groq_provider.generate(
            messages=[{"role": "user", "content": "Hi"}]
        )

        assert isinstance(response, ProviderResponse)
        assert response.text == "Hello there!"
        assert response.provider == "groq"
        assert response.model == "llama-3.3-70b-versatile"
        assert response.tokens_used == 15
        assert response.latency_ms > 0
        assert response.is_fallback is False

    @pytest.mark.asyncio
    async def test_generate_with_config(self, groq_provider):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_mock_completion())
        groq_provider._client = mock_client

        config = GenerationConfig(max_tokens=500, temperature=0.3)
        await groq_provider.generate(
            messages=[{"role": "user", "content": "Hi"}],
            config=config,
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_generate_api_error_propagates(self, groq_provider):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        groq_provider._client = mock_client

        with pytest.raises(Exception, match="API Error"):
            await groq_provider.generate(messages=[{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_generate_metadata_includes_token_counts(self, groq_provider):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("Hi", 10)
        )
        groq_provider._client = mock_client

        response = await groq_provider.generate(
            messages=[{"role": "user", "content": "Hi"}]
        )

        assert "finish_reason" in response.metadata
        assert "prompt_tokens" in response.metadata
        assert "completion_tokens" in response.metadata


class TestGroqProviderStream:
    """Tests for GroqProvider.generate_stream()."""

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, groq_provider):
        chunks = _mock_stream_chunks(["Hello", " world", "!"])

        mock_client = AsyncMock()

        async def mock_stream(*args, **kwargs):
            class MockAsyncIterator:
                def __init__(self, items):
                    self._items = iter(items)

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    try:
                        return next(self._items)
                    except StopIteration:
                        raise StopAsyncIteration from None

            return MockAsyncIterator(chunks)

        mock_client.chat.completions.create = mock_stream
        groq_provider._client = mock_client

        tokens = []
        async for token in groq_provider.generate_stream(
            messages=[{"role": "user", "content": "Hi"}]
        ):
            tokens.append(token)

        assert tokens == ["Hello", " world", "!"]


class TestGroqProviderHealth:
    """Tests for GroqProvider.health_check()."""

    @pytest.mark.asyncio
    async def test_healthy(self, groq_provider):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("hi", 2)
        )
        groq_provider._client = mock_client

        assert await groq_provider.health_check() is True

    @pytest.mark.asyncio
    async def test_unhealthy(self, groq_provider):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Timeout")
        )
        groq_provider._client = mock_client

        assert await groq_provider.health_check() is False


class TestGroqProviderImportError:
    """Tests for import error handling."""

    def test_missing_groq_package(self):
        import sys

        provider = GroqProvider(api_key="test-key")
        provider._client = None  # force re-init

        # Temporarily remove groq from sys.modules so the import re-fires
        groq_modules = {k: v for k, v in sys.modules.items() if k.startswith("groq")}
        for k in groq_modules:
            del sys.modules[k]

        real_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "groq":
                raise ImportError("No module named 'groq'")
            return real_import(name, *args, **kwargs)

        try:
            with (
                patch("builtins.__import__", side_effect=mock_import),
                pytest.raises(ImportError, match="groq package is required"),
            ):
                provider._get_client()
        finally:
            # Restore groq modules
            sys.modules.update(groq_modules)


class TestGroqProviderStreamError:
    """Tests for streaming error handling."""

    @pytest.mark.asyncio
    async def test_stream_error_propagates(self, groq_provider):
        mock_client = AsyncMock()

        async def mock_failing_stream(*args, **kwargs):
            raise ConnectionError("Stream interrupted")

        mock_client.chat.completions.create = mock_failing_stream
        groq_provider._client = mock_client

        with pytest.raises(ConnectionError, match="Stream interrupted"):
            async for _ in groq_provider.generate_stream(
                messages=[{"role": "user", "content": "Hi"}]
            ):
                pass


class TestGroqProviderGptOss:
    """gpt-oss reasoning headroom and retired-model fallback (mocked)."""

    @pytest.mark.asyncio
    async def test_reasoning_models_get_headroom_and_low_effort(self, monkeypatch):
        monkeypatch.delenv("GROQ_MODEL", raising=False)
        provider = GroqProvider(api_key="k")  # default: gpt-oss-120b
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_mock_completion())
        provider._client = mock_client

        await provider.generate(
            messages=[{"role": "user", "content": "Hi"}],
            config=GenerationConfig(max_tokens=100),
        )

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "openai/gpt-oss-120b"
        assert kwargs["max_tokens"] == 100 + 512
        assert kwargs["extra_body"] == {"reasoning_effort": "low"}

    @pytest.mark.asyncio
    async def test_retired_model_falls_back_once(self, monkeypatch):
        monkeypatch.delenv("GROQ_MODEL", raising=False)
        provider = GroqProvider(api_key="k")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("model_decommissioned: this model has been decommissioned"),
                _mock_completion("from fallback", 12),
            ]
        )
        provider._client = mock_client

        response = await provider.generate(messages=[{"role": "user", "content": "Hi"}])

        assert response.text == "from fallback"
        assert response.model == "openai/gpt-oss-20b"  # reports the model actually used
        calls = mock_client.chat.completions.create.call_args_list
        assert [c.kwargs["model"] for c in calls] == [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ]

    @pytest.mark.asyncio
    async def test_rate_limit_does_not_downgrade_the_model(self, monkeypatch):
        """A throttled judge must fail loudly, not silently switch to a weaker one."""
        monkeypatch.delenv("GROQ_MODEL", raising=False)
        provider = GroqProvider(api_key="k")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("429 rate limit reached")
        )
        provider._client = mock_client

        with pytest.raises(Exception, match="rate limit"):
            await provider.generate(messages=[{"role": "user", "content": "Hi"}])
        assert mock_client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_stream_falls_back_when_model_retired(self, monkeypatch):
        monkeypatch.delenv("GROQ_MODEL", raising=False)
        provider = GroqProvider(api_key="k")

        async def _gen():
            for chunk in _mock_stream_chunks(["A", "B"]):
                yield chunk

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[Exception("model_not_found"), _gen()]
        )
        provider._client = mock_client

        out = [t async for t in provider.generate_stream([{"role": "user", "content": "Hi"}])]

        assert out == ["A", "B"]
        models = [c.kwargs["model"] for c in mock_client.chat.completions.create.call_args_list]
        assert models == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
