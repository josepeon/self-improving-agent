"""Groq API provider — wraps the Groq SDK for remote inference (gpt-oss)."""

import contextlib
import logging
import os
import time
from collections.abc import AsyncGenerator
from typing import Any

from self_improving_agent.observability.cost_tracker import record_usage
from self_improving_agent.providers.base import (
    BaseModelProvider,
    GenerationConfig,
    ProviderResponse,
)
from self_improving_agent.providers.groq_models import (
    DEFAULT_GROQ_FALLBACK_MODEL,
    DEFAULT_GROQ_MODEL,
    is_retired_model_error,
    request_opts,
    resolve_groq_fallback_model,
    resolve_groq_model,
)

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_GROQ_FALLBACK_MODEL", "DEFAULT_GROQ_MODEL", "GroqProvider"]

DEFAULT_TIMEOUT = 30.0


class GroqProvider(BaseModelProvider):
    """Groq API provider for high-capability remote inference.

    Uses the Groq SDK to call hosted models (gpt-oss by default). Intended
    as the high-quality fallback when the local MLX model is insufficient.
    If the configured model has been retired by Groq, a call transparently
    retries once on the fallback model (rate limits are NOT retried this way,
    so a throttled judge never silently downgrades to a weaker one).

    Configuration via environment variables:
        GROQ_API_KEY: API key (required)
        GROQ_MODEL: Model name (default: openai/gpt-oss-120b)
        GROQ_FALLBACK_MODEL: Used if the model is retired (default: openai/gpt-oss-20b)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        fallback_model: str | None = None,
    ):
        self._api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._model: str = model or resolve_groq_model()
        self._fallback_model: str = fallback_model or resolve_groq_fallback_model()
        self._timeout = timeout
        self._client: Any = None

    @property
    def name(self) -> str:
        return "groq"

    def _get_client(self) -> Any:
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

    async def _create(
        self,
        client: Any,
        model: str,
        messages: list[dict[str, str]],
        config: GenerationConfig,
        *,
        stream: bool = False,
    ) -> Any:
        """One chat-completions call with model-specific request options."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stop": config.stop_sequences,
            **request_opts(model, config.max_tokens),
        }
        if stream:
            kwargs["stream"] = True
        return await client.chat.completions.create(**kwargs)

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
        model = self._model
        try:
            try:
                response = await self._create(client, model, messages, config)
            except Exception as e:
                if model != self._fallback_model and is_retired_model_error(e):
                    logger.warning(
                        "Groq model %s is retired (%s); retrying on %s",
                        model,
                        e,
                        self._fallback_model,
                    )
                    model = self._fallback_model
                    response = await self._create(client, model, messages, config)
                else:
                    raise
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
            with contextlib.suppress(Exception):
                record_usage(
                    provider="groq",
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

        logger.debug(
            "Groq response: model=%s tokens=%d latency=%.1fms",
            model,
            tokens_used,
            latency_ms,
        )

        return ProviderResponse(
            text=text,
            provider=self.name,
            model=model,
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
            try:
                stream = await self._create(
                    client, self._model, messages, config, stream=True
                )
            except Exception as e:
                if self._model != self._fallback_model and is_retired_model_error(e):
                    logger.warning(
                        "Groq model %s is retired (%s); streaming on %s",
                        self._model,
                        e,
                        self._fallback_model,
                    )
                    stream = await self._create(
                        client, self._fallback_model, messages, config, stream=True
                    )
                else:
                    raise

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
                **request_opts(self._model, 1),
            )
            return bool(response.choices)
        except Exception as e:
            logger.warning("Groq health check failed: %s", e)
            return False
