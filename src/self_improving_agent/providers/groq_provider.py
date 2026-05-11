"""Groq API provider — wraps Groq SDK for remote Llama 3.3 70B inference."""

import logging
import os
import time
from collections.abc import AsyncGenerator

from self_improving_agent.observability.cost_tracker import record_usage
from self_improving_agent.providers.base import (
    BaseModelProvider,
    GenerationConfig,
    ProviderResponse,
)

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TIMEOUT = 30.0


class GroqProvider(BaseModelProvider):
    """Groq API provider for high-capability remote inference.

    Uses the Groq SDK to call hosted Llama models. Intended as the
    high-quality fallback when the local MLX model is insufficient.

    Configuration via environment variables:
        GROQ_API_KEY: API key (required)
        GROQ_MODEL: Model name (default: llama-3.3-70b-versatile)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self._api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._model = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self._timeout = timeout
        self._client = None

    @property
    def name(self) -> str:
        return "groq"

    def _get_client(self):
        """Lazy-initialize the Groq client."""
        if self._client is None:
            try:
                from groq import AsyncGroq
            except ImportError as e:
                raise ImportError(
                    "groq package is required for GroqProvider. "
                    "Install with: pip install groq"
                ) from e

            if not self._api_key:
                raise ValueError(
                    "GROQ_API_KEY is required. Set it as an environment variable "
                    "or pass api_key to GroqProvider()."
                )

            self._client = AsyncGroq(
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    async def generate(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> ProviderResponse:
        """Generate a complete response via Groq API.

        Args:
            messages: Chat messages in OpenAI format.
            config: Generation parameters.

        Returns:
            ProviderResponse with text and metadata.
        """
        config = config or GenerationConfig()
        client = self._get_client()

        start = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop=config.stop_sequences,
            )
        except Exception as e:
            logger.error("Groq API call failed: %s", e)
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        choice = response.choices[0]
        text = choice.message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0

        # Record for the per-run cost tracker. No-op if observability isn't used.
        # ``record_usage`` reads the active ``attribute()`` context, so callers
        # that scoped their call get correct role attribution; the rest land
        # in "unknown".
        if response.usage is not None:
            try:
                record_usage(
                    provider="groq",
                    model=self._model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            except Exception:  # noqa: BLE001 — cost tracking must never break inference
                pass

        logger.debug(
            "Groq response: model=%s tokens=%d latency=%.1fms",
            self._model,
            tokens_used,
            latency_ms,
        )

        return ProviderResponse(
            text=text,
            provider=self.name,
            model=self._model,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            metadata={
                "finish_reason": choice.finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens from Groq API.

        Args:
            messages: Chat messages in OpenAI format.
            config: Generation parameters.

        Yields:
            Individual tokens/chunks as strings.
        """
        config = config or GenerationConfig()
        client = self._get_client()

        try:
            stream = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop=config.stop_sequences,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error("Groq streaming failed: %s", e)
            raise

    async def health_check(self) -> bool:
        """Check Groq API availability with a minimal request.

        Returns:
            True if the API responds successfully.
        """
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            return bool(response.choices)
        except Exception as e:
            logger.warning("Groq health check failed: %s", e)
            return False
