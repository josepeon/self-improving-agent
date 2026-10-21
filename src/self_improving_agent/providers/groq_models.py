"""Groq model defaults and request helpers, shared by the provider and scripts.

Groq retires hosted models with little notice. On 2026-08-16 it shut down
every Llama chat model (``llama-3.3-70b-versatile``, ``llama-3.1-8b-instant``)
and ``qwen/qwen3-32b``; the current production chat models are the
``openai/gpt-oss`` family. Keeping the defaults, the stale-pin guard, and
the request quirks in one place means the next retirement is a one-line
change instead of a hunt through scripts.
"""

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GROQ_FALLBACK_MODEL = "openai/gpt-oss-20b"

# Model ids known to be retired. A GROQ_MODEL / GROQ_FALLBACK_MODEL env pin
# naming one of these (typically a stale .env copied from an old example) is
# ignored with a warning, because it can only ever fail.
RETIRED_GROQ_MODELS = frozenset(
    {
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-specdec",
        "qwen/qwen3-32b",
    }
)

# gpt-oss models spend completion tokens on internal reasoning before the
# visible answer, so max_tokens must leave room for it or the reply comes
# back empty/truncated.
REASONING_HEADROOM_TOKENS = 512

_RETIRED_ERROR_MARKERS = (
    "decommissioned",
    "model_not_found",
    "model not found",
    "does not exist",
)


def _live_model(env_var: str, default: str) -> str:
    value = (os.getenv(env_var) or "").strip()
    if not value:
        return default
    if value in RETIRED_GROQ_MODELS:
        logger.warning(
            "%s=%s was retired by Groq; using %s instead. Update your environment.",
            env_var,
            value,
            default,
        )
        return default
    return value


def resolve_groq_model() -> str:
    """Primary Groq model: ``GROQ_MODEL`` if live, else the default."""
    return _live_model("GROQ_MODEL", DEFAULT_GROQ_MODEL)


def resolve_groq_fallback_model() -> str:
    """Fallback Groq model: ``GROQ_FALLBACK_MODEL`` if live, else the default."""
    return _live_model("GROQ_FALLBACK_MODEL", DEFAULT_GROQ_FALLBACK_MODEL)


def is_reasoning_model(model: str) -> bool:
    return model.startswith("openai/gpt-oss")


def request_opts(model: str, max_tokens: int) -> dict:
    """Keyword args for ``chat.completions.create`` that are model-specific.

    Returns ``max_tokens`` (padded for reasoning models) and, for reasoning
    models, a low ``reasoning_effort`` so replies stay fast and cheap.
    """
    if is_reasoning_model(model):
        return {
            "max_tokens": max_tokens + REASONING_HEADROOM_TOKENS,
            "extra_body": {"reasoning_effort": "low"},
        }
    return {"max_tokens": max_tokens}


def is_retired_model_error(exc: BaseException) -> bool:
    """True when an API error means the model itself is gone (not a rate limit)."""
    text = str(exc).lower()
    return any(marker in text for marker in _RETIRED_ERROR_MARKERS)
