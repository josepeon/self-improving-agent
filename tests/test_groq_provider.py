"""Tests for GroqProvider (mocked — no real API calls)."""

from unittest.mock import AsyncMock, MagicMock

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

    def test_default_model(self):
        provider = GroqProvider(api_key="key")
        assert provider._model == "llama-3.3-70b-versatile"

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
