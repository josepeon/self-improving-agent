"""Tests for ModelRouter."""

from unittest.mock import AsyncMock

import pytest

from self_improving_agent.providers.base import ProviderResponse
from self_improving_agent.providers.router import ModelRouter


def _make_response(provider: str = "test", text: str = "Hello") -> ProviderResponse:
    """Create a test ProviderResponse."""
    return ProviderResponse(
        text=text,
        provider=provider,
        model=f"{provider}-model",
        latency_ms=50.0,
        tokens_used=10,
    )


@pytest.fixture
def mock_local():
    """Create a mock local provider."""
    provider = AsyncMock()
    provider.name = "mlx"
    provider.generate = AsyncMock(return_value=_make_response("mlx", "Local response"))
    provider.generate_stream = AsyncMock()
    provider.health_check = AsyncMock(return_value=True)
    return provider


@pytest.fixture
def mock_remote():
    """Create a mock remote provider."""
    provider = AsyncMock()
    provider.name = "groq"
    provider.generate = AsyncMock(
        return_value=_make_response("groq", "Remote response")
    )
    provider.generate_stream = AsyncMock()
    provider.health_check = AsyncMock(return_value=True)
    return provider


@pytest.fixture
def router(mock_local, mock_remote):
    """Create a ModelRouter with mock providers."""
    return ModelRouter(local=mock_local, remote=mock_remote)


class TestRouterForceProvider:
    """Tests for explicit force_provider routing."""

    @pytest.mark.asyncio
    async def test_force_local(self, router, mock_local):
        response = await router.route(
            messages=[{"role": "user", "content": "Hi"}],
            force_provider="local",
        )
        assert response.provider == "mlx"
        mock_local.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_force_remote(self, router, mock_remote):
        response = await router.route(
            messages=[{"role": "user", "content": "Hi"}],
            force_provider="remote",
        )
        assert response.provider == "groq"
        mock_remote.generate.assert_called_once()


class TestRouterQualityMode:
    """Tests for quality_mode routing."""

    @pytest.mark.asyncio
    async def test_fast_mode_routes_local(self, router, mock_local):
        response = await router.route(
            messages=[{"role": "user", "content": "Hi"}],
            quality_mode="fast",
        )
        assert response.provider == "mlx"

    @pytest.mark.asyncio
    async def test_quality_mode_routes_remote(self, router, mock_remote):
        response = await router.route(
            messages=[{"role": "user", "content": "Hi"}],
            quality_mode="quality",
        )
        assert response.provider == "groq"


class TestRouterComplexityDetection:
    """Tests for complexity-based routing."""

    @pytest.mark.asyncio
    async def test_simple_message_routes_local(self, router, mock_local):
        response = await router.route(
            messages=[{"role": "user", "content": "Hello!"}],
        )
        assert response.provider == "mlx"

    @pytest.mark.asyncio
    async def test_complex_message_routes_remote(self, router, mock_remote):
        response = await router.route(
            messages=[
                {
                    "role": "user",
                    "content": "Explain how neural networks work step by step",
                }
            ],
        )
        assert response.provider == "groq"

    @pytest.mark.asyncio
    async def test_code_request_routes_remote(self, router, mock_remote):
        response = await router.route(
            messages=[
                {"role": "user", "content": "Write code for a sorting algorithm"}
            ],
        )
        assert response.provider == "groq"

    @pytest.mark.asyncio
    async def test_debug_request_routes_remote(self, router, mock_remote):
        response = await router.route(
            messages=[
                {"role": "user", "content": "Debug this error in my Python script"}
            ],
        )
        assert response.provider == "groq"


class TestRouterFallback:
    """Tests for automatic fallback behavior."""

    @pytest.mark.asyncio
    async def test_fallback_on_local_failure(self, router, mock_local, mock_remote):
        mock_local.generate = AsyncMock(side_effect=Exception("Model crashed"))

        response = await router.route(
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert response.provider == "groq"
        assert response.is_fallback is True

    @pytest.mark.asyncio
    async def test_fallback_increments_counter(self, router, mock_local):
        mock_local.generate = AsyncMock(side_effect=Exception("Error"))

        await router.route(messages=[{"role": "user", "content": "Hi"}])

        assert router.stats["fallback_count"] == 1


class TestRouterStats:
    """Tests for routing statistics."""

    @pytest.mark.asyncio
    async def test_initial_stats(self, router):
        stats = router.stats
        assert stats["total_requests"] == 0
        assert stats["local_requests"] == 0
        assert stats["remote_requests"] == 0
        assert stats["fallback_count"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_requests(self, router):
        # Simple request → local
        await router.route(messages=[{"role": "user", "content": "Hi"}])
        # Complex request → remote
        await router.route(
            messages=[{"role": "user", "content": "Explain quantum physics"}]
        )

        stats = router.stats
        assert stats["total_requests"] == 2
        assert stats["local_requests"] == 1
        assert stats["remote_requests"] == 1
        assert stats["fallback_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_stats_with_fallback(self, router, mock_local):
        mock_local.generate = AsyncMock(side_effect=Exception("Error"))

        await router.route(messages=[{"role": "user", "content": "Hi"}])

        stats = router.stats
        assert stats["total_requests"] == 1
        assert stats["local_requests"] == 1
        assert stats["fallback_count"] == 1
        assert stats["fallback_rate"] == 1.0


class TestRouterComplexityAssessment:
    """Tests for _assess_complexity internal method."""

    def test_empty_messages(self, router):
        assert router._assess_complexity([]) is False

    def test_simple_message(self, router):
        assert (
            router._assess_complexity([{"role": "user", "content": "Hello"}]) is False
        )

    def test_complex_markers(self, router):
        complex_messages = [
            "Explain the theory of relativity",
            "Analyze this dataset",
            "Compare Python and Rust",
            "Write code for a web scraper",
            "Step by step guide to cooking pasta",
        ]

        for msg in complex_messages:
            assert (
                router._assess_complexity([{"role": "user", "content": msg}]) is True
            ), f"Expected '{msg}' to be complex"


class TestRouterStream:
    """Tests for route_stream()."""

    @pytest.mark.asyncio
    async def test_stream_simple_routes_local(self, router, mock_local):
        async def _fake_stream(*args, **kwargs):
            for t in ["Hi", " there"]:
                yield t

        mock_local.generate_stream = _fake_stream

        tokens = []
        async for token in router.route_stream(
            messages=[{"role": "user", "content": "Hi"}],
        ):
            tokens.append(token)

        assert tokens == ["Hi", " there"]

    @pytest.mark.asyncio
    async def test_stream_complex_routes_remote(self, router, mock_remote):
        async def _fake_stream(*args, **kwargs):
            for t in ["Sure", ", let me", " explain"]:
                yield t

        mock_remote.generate_stream = _fake_stream

        tokens = []
        async for token in router.route_stream(
            messages=[{"role": "user", "content": "Explain how GPUs work"}],
        ):
            tokens.append(token)

        assert tokens == ["Sure", ", let me", " explain"]

    @pytest.mark.asyncio
    async def test_stream_force_remote(self, router, mock_remote):
        async def _fake_stream(*args, **kwargs):
            yield "forced"

        mock_remote.generate_stream = _fake_stream

        tokens = []
        async for token in router.route_stream(
            messages=[{"role": "user", "content": "Hi"}],
            force_provider="remote",
        ):
            tokens.append(token)

        assert tokens == ["forced"]

    @pytest.mark.asyncio
    async def test_stream_force_local(self, router, mock_local):
        async def _fake_stream(*args, **kwargs):
            yield "local"

        mock_local.generate_stream = _fake_stream

        tokens = []
        async for token in router.route_stream(
            messages=[{"role": "user", "content": "Hi"}],
            force_provider="local",
        ):
            tokens.append(token)

        assert tokens == ["local"]

    @pytest.mark.asyncio
    async def test_stream_quality_mode(self, router, mock_remote):
        async def _fake_stream(*args, **kwargs):
            yield "quality"

        mock_remote.generate_stream = _fake_stream

        tokens = []
        async for token in router.route_stream(
            messages=[{"role": "user", "content": "Hi"}],
            quality_mode="quality",
        ):
            tokens.append(token)

        assert tokens == ["quality"]

    @pytest.mark.asyncio
    async def test_stream_fast_mode(self, router, mock_local):
        async def _fake_stream(*args, **kwargs):
            yield "fast"

        mock_local.generate_stream = _fake_stream

        tokens = []
        async for token in router.route_stream(
            messages=[{"role": "user", "content": "Hi"}],
            quality_mode="fast",
        ):
            tokens.append(token)

        assert tokens == ["fast"]

    @pytest.mark.asyncio
    async def test_stream_fallback_on_local_failure(
        self, router, mock_local, mock_remote
    ):
        async def _fail_stream(*args, **kwargs):
            raise RuntimeError("Local stream crashed")
            yield

        async def _remote_stream(*args, **kwargs):
            yield "fallback"

        mock_local.generate_stream = _fail_stream
        mock_remote.generate_stream = _remote_stream

        tokens = []
        async for token in router.route_stream(
            messages=[{"role": "user", "content": "Hi"}],
        ):
            tokens.append(token)

        assert tokens == ["fallback"]
        assert router.stats["fallback_count"] == 1

    @pytest.mark.asyncio
    async def test_stream_remote_failure_raises(self, router, mock_remote):
        async def _fail_stream(*args, **kwargs):
            raise RuntimeError("Remote down")
            yield

        mock_remote.generate_stream = _fail_stream

        with pytest.raises(RuntimeError, match="Remote down"):
            async for _ in router.route_stream(
                messages=[{"role": "user", "content": "Explain quantum physics"}],
            ):
                pass
