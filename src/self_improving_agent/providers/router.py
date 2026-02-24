"""Intelligent model router — routes requests to the optimal provider."""

import logging
from typing import ClassVar

from self_improving_agent.providers.base import (
    BaseModelProvider,
    GenerationConfig,
    ProviderResponse,
)

logger = logging.getLogger(__name__)


class ModelRouter:
    """Routes requests to the optimal provider based on complexity analysis.

    Routing logic (in priority order):
    1. Explicit force_provider overrides everything.
    2. quality_mode "fast" → local, "quality" → remote.
    3. Complexity markers in the user message → remote.
    4. Default: local model with automatic fallback to remote on failure.

    The router tracks performance metrics to inform future routing decisions.
    """

    # Keywords that suggest the request needs a more capable model
    COMPLEXITY_MARKERS: ClassVar[list[str]] = [
        "explain",
        "analyze",
        "compare",
        "why",
        "how does",
        "step by step",
        "in detail",
        "pros and cons",
        "write code",
        "debug",
        "calculate",
        "summarize",
        "translate",
        "evaluate",
    ]

    def __init__(
        self,
        local: BaseModelProvider,
        remote: BaseModelProvider,
    ):
        self.local = local
        self.remote = remote

        # Performance tracking
        self._local_requests = 0
        self._remote_requests = 0
        self._fallback_count = 0
        self._total_requests = 0

    @property
    def stats(self) -> dict:
        """Current routing statistics."""
        return {
            "total_requests": self._total_requests,
            "local_requests": self._local_requests,
            "remote_requests": self._remote_requests,
            "fallback_count": self._fallback_count,
            "fallback_rate": (
                self._fallback_count / self._total_requests
                if self._total_requests > 0
                else 0.0
            ),
            "local_ratio": (
                self._local_requests / self._total_requests
                if self._total_requests > 0
                else 0.0
            ),
        }

    def _assess_complexity(self, messages: list[dict[str, str]]) -> bool:
        """Determine if the request requires the remote (more capable) model.

        Args:
            messages: Chat messages in OpenAI format.

        Returns:
            True if the request appears complex.
        """
        if not messages:
            return False

        user_msg = messages[-1].get("content", "").lower()
        return any(marker in user_msg for marker in self.COMPLEXITY_MARKERS)

    async def route(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
        quality_mode: str = "default",
        force_provider: str | None = None,
    ) -> ProviderResponse:
        """Route a request to the optimal provider.

        Args:
            messages: Chat messages in OpenAI format.
            config: Generation parameters.
            quality_mode: "fast" (local), "quality" (remote), or "default" (auto).
            force_provider: Override routing with "local" or "remote".

        Returns:
            ProviderResponse from the selected provider.
        """
        self._total_requests += 1

        # 1. Explicit override
        if force_provider == "remote":
            logger.debug("Forced routing to remote provider")
            self._remote_requests += 1
            return await self.remote.generate(messages, config)
        if force_provider == "local":
            logger.debug("Forced routing to local provider")
            self._local_requests += 1
            return await self.local.generate(messages, config)

        # 2. Quality mode
        if quality_mode == "fast":
            logger.debug("Quality mode 'fast' → local provider")
            self._local_requests += 1
            return await self._try_local_with_fallback(messages, config)
        if quality_mode == "quality":
            logger.debug("Quality mode 'quality' → remote provider")
            self._remote_requests += 1
            return await self.remote.generate(messages, config)

        # 3. Complexity-based routing
        if self._assess_complexity(messages):
            logger.debug("Complex request detected → remote provider")
            self._remote_requests += 1
            return await self.remote.generate(messages, config)

        # 4. Default: local with fallback
        logger.debug("Default routing → local provider with fallback")
        self._local_requests += 1
        return await self._try_local_with_fallback(messages, config)

    async def _try_local_with_fallback(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> ProviderResponse:
        """Try local provider, fall back to remote on failure.

        Args:
            messages: Chat messages.
            config: Generation parameters.

        Returns:
            ProviderResponse from whichever provider succeeds.
        """
        try:
            return await self.local.generate(messages, config)
        except Exception as e:
            logger.warning(
                "Local provider failed (%s), falling back to remote: %s",
                self.local.name,
                e,
            )
            self._fallback_count += 1
            response = await self.remote.generate(messages, config)
            response.is_fallback = True
            return response

    async def route_stream(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
        quality_mode: str = "default",
        force_provider: str | None = None,
    ):
        """Route a streaming request to the optimal provider.

        Same routing logic as route(), but returns an async generator.
        """
        self._total_requests += 1

        # Determine provider
        if force_provider == "remote":
            provider = self.remote
        elif force_provider == "local":
            provider = self.local
        elif quality_mode == "quality":
            provider = self.remote
        elif quality_mode == "fast":
            provider = self.local
        elif self._assess_complexity(messages):
            provider = self.remote
        else:
            provider = self.local

        if provider == self.local:
            self._local_requests += 1
        else:
            self._remote_requests += 1

        try:
            async for token in provider.generate_stream(messages, config):
                yield token
        except Exception as e:
            if provider == self.local:
                logger.warning("Local streaming failed, falling back to remote: %s", e)
                self._fallback_count += 1
                async for token in self.remote.generate_stream(messages, config):
                    yield token
            else:
                raise
