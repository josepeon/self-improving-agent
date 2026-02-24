"""Logging middleware — decorator that captures LLM interactions.

Wraps async generate functions to transparently log inputs, outputs,
timing, and errors to an :class:`InteractionLogStore`.

Usage::

    store = InteractionLogStore()

    @log_llm_interaction(store)
    async def generate(message: str, *, quality_mode: str = "default") -> ProviderResponse:
        ...
"""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from self_improving_agent.logging.interaction_log import InteractionLog
from self_improving_agent.logging.storage import InteractionLogStore


def log_llm_interaction(
    store: InteractionLogStore,
    *,
    extract_message: Callable[..., str] | None = None,
    extract_system_prompt: Callable[..., str] | None = None,
) -> Callable[..., Callable[..., Any]]:
    """Decorator factory that logs every call to the wrapped async function.

    Parameters
    ----------
    store:
        The :class:`InteractionLogStore` to write logs to.
    extract_message:
        Optional callable ``(args, kwargs) -> str`` to pull the user
        message from the decorated function's arguments.  Defaults to
        ``kwargs["message"]`` or ``args[0]``.
    extract_system_prompt:
        Optional callable ``(args, kwargs) -> str`` to pull the system
        prompt.  Defaults to ``kwargs.get("system_prompt", "")``.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = InteractionLog()

            # ── Capture inputs ────────────────────────────────
            if extract_message is not None:
                log.user_message = extract_message(args, kwargs)
            else:
                log.user_message = _default_extract_message(args, kwargs)

            if extract_system_prompt is not None:
                log.system_prompt = extract_system_prompt(args, kwargs)
            else:
                log.system_prompt = kwargs.get("system_prompt", "")

            log.system_prompt_version = kwargs.get("prompt_version", "")
            log.quality_mode = kwargs.get("quality_mode", "default")
            log.request_id = kwargs.get("request_id", "")
            log.user_id = kwargs.get("user_id", "")
            log.session_id = kwargs.get("session_id", "")

            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)

                # ── Capture outputs from ProviderResponse ─────
                log.response_text = getattr(result, "text", str(result))
                log.provider = getattr(result, "provider", "")
                log.model_name = getattr(result, "model", "")
                log.is_fallback = getattr(result, "is_fallback", False)
                log.response_tokens = getattr(result, "tokens_used", 0)

                return result

            except Exception as exc:
                log.failure_category = "error"
                log.response_text = f"{type(exc).__name__}: {exc}"
                raise

            finally:
                log.latency_ms = (time.perf_counter() - start) * 1000
                await store.log_interaction(log)

        return wrapper

    return decorator


def _default_extract_message(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Best-effort extraction of the user message from function arguments."""
    if "message" in kwargs:
        return str(kwargs["message"])
    if "prompt" in kwargs:
        return str(kwargs["prompt"])
    if args:
        return str(args[0])
    return ""
