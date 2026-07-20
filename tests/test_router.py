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
    async def test_code_request_with_density_routes_remote(self, router, mock_remote):
        # Heuristic now considers code-density signals, not just keywords.
        # Triggers: 'write code' marker (0.3) + code fence (0.3) + identifiers
        # → ≥ threshold (0.5).
        response = await router.route(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Write code for a sorting algorithm. Here's the "
                        "signature:\n```python\n"
                        "def sort_users(users: list[User]) -> list[User]:\n"
                        "    pass\n"
                        "```\n"
                    ),
                }
            ],
        )
        assert response.provider == "groq"

    @pytest.mark.asyncio
    async def test_debug_request_with_paths_routes_remote(self, router, mock_remote):
        # 'debug' (0.3) + path mentions + length pushes over threshold.
        response = await router.route(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Debug this error in src/auth/handler.py:\n"
                        "tests/test_handler.py is failing with an "
                        "AttributeError on UserService.authenticate when "
                        "tokens.refresh returns None. Step by step please."
                    ),
                }
            ],
        )
        assert response.provider == "groq"

    @pytest.mark.asyncio
    async def test_short_explain_stays_local(self, router, mock_local):
        # New behaviour: a short 'Explain X' that an 8B handles fine should
        # NOT be routed remote.
        response = await router.route(
            messages=[{"role": "user", "content": "Explain recursion"}],
        )
        assert response.provider == "mlx"


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
        # Complex request → remote (need enough signal to clear threshold)
        await router.route(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Explain step by step in detail how to compare and "
                        "evaluate the pros and cons of microservice vs monolith "
                        "architectures, including code organisation tradeoffs."
                    ),
                }
            ]
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

    def test_strong_signals_route_remote(self, router):
        """Requests with enough cumulative signal cross the 0.5 threshold."""
        strong_signals = [
            # Multi-marker (step by step = 0.4 + design = 0.3 + length)
            "Step by step, design a rate limiter that handles bursts.",
            # Code fence + 'write code' marker + identifiers
            "Write code that does this:\n```python\n"
            "from auth.handlers import UserService\n"
            "user_service.authenticate(token)\n```",
            # Long + reasoning
            "Compare and contrast the architectural tradeoffs of "
            "microservices versus a modular monolith in detail, "
            "considering ops burden, fault isolation, and team scaling.",
        ]
        for msg in strong_signals:
            assert (
                router._assess_complexity([{"role": "user", "content": msg}]) is True
            ), f"Expected '{msg[:60]}...' to be complex"

    def test_short_marker_alone_stays_local(self, router):
        """A single marker on a short prompt should NOT trigger remote.

        This is the intentional behaviour change in B2: 'any keyword = remote'
        was over-eager; an 8B local model handles 'Explain X' fine.
        """
        for msg in ["Explain X", "Analyze this", "Compare A and B"]:
            assert (
                router._assess_complexity([{"role": "user", "content": msg}]) is False
            ), f"Expected '{msg}' to stay local"

    def test_complexity_score_bounded(self, router):
        # Even a worst-case input doesn't exceed 1.0
        worst_case = (
            "Explain in detail step by step why analyzing this design "
            "compares with debugging code and evaluating pros and cons. "
            "```python\nfrom a.b import c\nclass MyClass: pass\n```\n"
        ) * 10
        score = router._complexity_score(
            [
                {"role": "user", "content": worst_case},
                {"role": "assistant", "content": "..."},
                {"role": "user", "content": "?"},
            ]
        )
        assert 0.0 <= score <= 1.0

    def test_threshold_env_override(self, monkeypatch, mock_local, mock_remote):
        # With threshold ~0, everything routes remote
        monkeypatch.setenv("COMPLEXITY_THRESHOLD", "0.01")
        from self_improving_agent.providers.router import ModelRouter

        r = ModelRouter(mock_local, mock_remote)
        assert (
            r._assess_complexity([{"role": "user", "content": "Hello"}]) is False
        )  # Score of 0 is still < 0.01
        assert (
            r._assess_complexity([{"role": "user", "content": "Explain X"}]) is True
        )  # Score of 0.2 > 0.01


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
            messages=[
                {
                    "role": "user",
                    "content": "Explain how neural networks work step by step",
                }
            ],
        ):
            tokens.append(token)

        assert tokens == ["Sure", ", let me", " explain"]
        # Must be a direct remote route, not a local failure + fallback.
        assert router.stats["remote_requests"] == 1
        assert router.stats["fallback_count"] == 0

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
                messages=[
                    {
                        "role": "user",
                        "content": "Explain how neural networks work step by step",
                    }
                ],
            ):
                pass
        # Confirm we actually routed remote (no fallback path involved).
        assert router.stats["remote_requests"] == 1
