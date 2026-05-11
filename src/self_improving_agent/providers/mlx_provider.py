"""MLX provider — local Llama inference on Apple Silicon.

Uses mlx-lm for fast inference and fine-tuning on M-series Macs.
Default model: Llama 3.1 8B Instruct 4-bit (~4.5GB, ~6GB RAM).
"""

import asyncio
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

DEFAULT_MLX_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"


def _make_sampler(temperature: float, top_p: float):
    """Build an mlx-lm sampler if the newer API is available; else return None.

    mlx-lm < 0.20 took temp= / top_p= as direct kwargs to generate().
    mlx-lm >= 0.20 dropped them and expects a sampler= callable. We
    detect at call time so the provider works on both versions.
    """
    try:
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        return None
    return make_sampler(temp=temperature, top_p=top_p)


class MLXProvider(BaseModelProvider):
    """Local MLX provider for Apple Silicon inference.

    Loads and runs Llama models locally using the MLX framework.
    Models are lazily downloaded on first use to ~/.cache/huggingface/.

    Configuration via environment variables:
        MLX_MODEL: Model path or HuggingFace ID
            (default: mlx-community/Meta-Llama-3.1-8B-Instruct-4bit)
    """

    def __init__(
        self,
        model_path: str | None = None,
        adapter_path: str | None = None,
    ):
        self._model_path = model_path or os.getenv("MLX_MODEL", DEFAULT_MLX_MODEL)
        self._adapter_path = adapter_path
        self._model = None
        self._tokenizer = None
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "mlx"

    async def _ensure_loaded(self) -> None:
        """Lazy-load model on first use (thread-safe)."""
        if self._model is not None:
            return

        async with self._lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return

            logger.info("Loading MLX model: %s", self._model_path)
            start = time.perf_counter()

            try:
                from mlx_lm import load
            except ImportError as e:
                raise ImportError(
                    "mlx-lm package is required for MLXProvider. "
                    "Install with: pip install mlx-lm"
                ) from e

            # Run the blocking load in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            self._model, self._tokenizer = await loop.run_in_executor(
                None,
                lambda: load(
                    self._model_path,
                    adapter_path=self._adapter_path,
                ),
            )

            elapsed = (time.perf_counter() - start) * 1000
            logger.info("MLX model loaded in %.1fms", elapsed)

    async def generate(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> ProviderResponse:
        """Generate a complete response using local MLX model.

        Args:
            messages: Chat messages in OpenAI format.
            config: Generation parameters.

        Returns:
            ProviderResponse with text and metadata.
        """
        await self._ensure_loaded()
        config = config or GenerationConfig()

        # Format messages using the model's chat template
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        start = time.perf_counter()
        try:
            from mlx_lm import generate as mlx_generate

            # mlx-lm renamed temp/top_p into a sampler in 0.20+. We build the
            # sampler when available and fall back to the legacy kwargs
            # signature when it isn't, so the provider works on both.
            sampler = _make_sampler(config.temperature, config.top_p)
            generate_kwargs: dict = {
                "model": self._model,
                "tokenizer": self._tokenizer,
                "prompt": prompt,
                "max_tokens": config.max_tokens,
            }
            if sampler is not None:
                generate_kwargs["sampler"] = sampler
            else:
                generate_kwargs["temp"] = config.temperature
                generate_kwargs["top_p"] = config.top_p

            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None,
                lambda: mlx_generate(**generate_kwargs),
            )
        except Exception as e:
            logger.error("MLX generation failed: %s", e)
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        prompt_tokens = len(self._tokenizer.encode(prompt))
        completion_tokens = len(self._tokenizer.encode(response_text))
        tokens_used = prompt_tokens + completion_tokens

        # Cost is $0 for local MLX but the call still counts towards token
        # totals — useful for budget-vs-latency analysis.
        try:
            record_usage(
                provider="mlx",
                model=self._model_path,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception:  # noqa: BLE001
            pass

        logger.debug(
            "MLX response: model=%s tokens=%d latency=%.1fms",
            self._model_path,
            tokens_used,
            latency_ms,
        )

        return ProviderResponse(
            text=response_text,
            provider=self.name,
            model=self._model_path,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            metadata={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens from local MLX model.

        Args:
            messages: Chat messages in OpenAI format.
            config: Generation parameters.

        Yields:
            Individual tokens/chunks as strings.
        """
        await self._ensure_loaded()
        config = config or GenerationConfig()

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        try:
            # mlx-lm 0.20+ exports stream_generate at the top level;
            # older versions had it under .utils. Try both.
            try:
                from mlx_lm import stream_generate
            except ImportError:
                from mlx_lm.utils import stream_generate

            # stream_generate is a synchronous generator — wrap it
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue[str | None] = asyncio.Queue()

            sampler = _make_sampler(config.temperature, config.top_p)
            stream_kwargs: dict = {
                "model": self._model,
                "tokenizer": self._tokenizer,
                "prompt": prompt,
                "max_tokens": config.max_tokens,
            }
            if sampler is not None:
                stream_kwargs["sampler"] = sampler
            else:
                stream_kwargs["temp"] = config.temperature

            def _stream_worker() -> None:
                try:
                    for response in stream_generate(**stream_kwargs):
                        # mlx-lm yields differently-shaped objects across
                        # versions: 0.20+ has .text, older versions had
                        # .token_string, very old yields raw strings.
                        if hasattr(response, "text"):
                            token = response.text
                        elif hasattr(response, "token_string"):
                            token = response.token_string
                        else:
                            token = str(response)
                        asyncio.run_coroutine_threadsafe(queue.put(token), loop)
                finally:
                    asyncio.run_coroutine_threadsafe(
                        queue.put(None),
                        loop,  # Sentinel
                    )

            loop.run_in_executor(None, _stream_worker)

            while True:
                token = await queue.get()
                if token is None:
                    break
                yield token

        except Exception as e:
            logger.error("MLX streaming failed: %s", e)
            raise

    async def health_check(self) -> bool:
        """Check if MLX model can generate.

        Returns:
            True if the model is loaded and can produce output.
        """
        try:
            await self._ensure_loaded()
            response = await self.generate(
                messages=[{"role": "user", "content": "hi"}],
                config=GenerationConfig(max_tokens=5, temperature=0.0),
            )
            return bool(response.text)
        except Exception as e:
            logger.warning("MLX health check failed: %s", e)
            return False
