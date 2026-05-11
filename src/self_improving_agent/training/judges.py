"""
Judge abstractions for self-play conversation filtering.

The self-play data generator (``scripts/generate_conversations.py``) uses
a single Groq 70B call to score each conversation. One judge has a
viewpoint — biases creep into the training data we keep.

This module introduces a small ``Judge`` protocol and an ``EnsembleJudge``
that combines multiple judges via median (or other aggregations). The
self-play generator instantiates an EnsembleJudge by default; the
existing single-judge behaviour is preserved by passing a one-element
panel.

Implementations:
  - ``GroqJudge``    — scores via Groq 70B (existing behaviour)
  - ``LocalJudge``   — scores via SIA's own current production model
                       (interesting self-judging signal)
  - ``EnsembleJudge`` — combines N judges, picks median by default

Calibration: each judge's running statistics (mean, count) live on the
instance so the dashboard can show which judges are strict/lax over time.
"""

from __future__ import annotations

import asyncio
import logging
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from self_improving_agent.providers.base import (
    BaseModelProvider,
    GenerationConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------


@dataclass
class JudgeVerdict:
    """One judge's verdict on a conversation."""

    judge: str
    score: float  # 0..10
    rationale: str = ""
    error: str | None = None


@dataclass
class EnsembleVerdict:
    """Aggregated verdict from a panel of judges."""

    score: float  # aggregate
    judges: list[JudgeVerdict]
    aggregation: str = "median"

    @property
    def kept(self) -> bool:
        """Convenience flag; the ensemble decides per its own threshold via
        ``EnsembleJudge.is_kept(verdict)``. We don't bake threshold here so
        the verdict object is reusable."""
        return False  # legacy, kept for compat — use EnsembleJudge.is_kept()


# ---------------------------------------------------------------------
# Judge protocol + impls
# ---------------------------------------------------------------------


class Judge(Protocol):
    """Anything that can score a conversation 0..10."""

    name: str

    async def score(
        self,
        conversation: list[dict[str, str]],
    ) -> JudgeVerdict: ...


_SCORE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\b")


def _parse_score(raw: str) -> float:
    """Pull the first 0..10 number out of an LLM judge response. Clamps."""
    match = _SCORE_RE.search(raw)
    if not match:
        return 0.0
    try:
        value = float(match.group(1))
    except ValueError:
        return 0.0
    return max(0.0, min(10.0, value))


_DEFAULT_SYSTEM = (
    "You are an expert evaluator scoring AI conversation quality on a "
    "0-10 scale. 10 = expert-level, helpful, accurate, well-structured. "
    "0 = unhelpful, wrong, or hallucinated. Respond with only the number, "
    "optionally followed by a brief one-line rationale."
)


def _format_conversation(conversation: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
        for m in conversation
    )


class ProviderJudge:
    """Judge backed by any BaseModelProvider — Groq, MLX, OpenAI, etc."""

    def __init__(
        self,
        provider: BaseModelProvider,
        *,
        name: str | None = None,
        system_prompt: str = _DEFAULT_SYSTEM,
        temperature: float = 0.3,
        max_tokens: int = 96,
    ) -> None:
        self.provider = provider
        self.name = name or f"{provider.name}-judge"
        self._system = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        # Calibration tracking
        self.scores_seen: list[float] = []

    async def score(self, conversation: list[dict[str, str]]) -> JudgeVerdict:
        prompt = (
            "Score this conversation:\n\n"
            + _format_conversation(conversation)
            + "\n\nScore (0-10):"
        )
        config = GenerationConfig(
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        try:
            response = await self.provider.generate(
                [
                    {"role": "system", "content": self._system},
                    {"role": "user", "content": prompt},
                ],
                config,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s judge failed: %s", self.name, exc)
            return JudgeVerdict(
                judge=self.name,
                score=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )

        score = _parse_score(response.text)
        self.scores_seen.append(score)
        return JudgeVerdict(
            judge=self.name,
            score=score,
            rationale=response.text.strip(),
        )

    @property
    def mean_score(self) -> float | None:
        return statistics.fmean(self.scores_seen) if self.scores_seen else None


def GroqJudge(  # noqa: N802 - factory function looks like a class
    api_key: str | None = None,
    model: str | None = None,
    name: str = "groq",
) -> ProviderJudge:
    """Convenience constructor for a Groq-backed judge."""
    from self_improving_agent.providers.groq_provider import GroqProvider

    return ProviderJudge(GroqProvider(api_key=api_key, model=model), name=name)


def LocalJudge(  # noqa: N802
    adapter_path: str | None = None,
    model_path: str | None = None,
    name: str = "local",
) -> ProviderJudge:
    """Convenience constructor for an MLX-backed local judge."""
    from self_improving_agent.providers.mlx_provider import MLXProvider

    return ProviderJudge(
        MLXProvider(model_path=model_path, adapter_path=adapter_path),
        name=name,
    )


# ---------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------


Aggregator = Callable[[list[float]], float]


def _median(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return statistics.median(scores)


def _mean(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return statistics.fmean(scores)


def _min(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return min(scores)


AGGREGATORS: dict[str, Aggregator] = {
    "median": _median,
    "mean": _mean,
    "min": _min,
}


@dataclass
class EnsembleJudge:
    """Combine N judges into a single verdict.

    Args:
        judges: Each judge must implement ``score(conversation)``.
        threshold: Aggregate score at-or-above this counts as 'kept'.
        aggregation: One of 'median' (default), 'mean', 'min'.
        max_parallel: Cap concurrent judge calls (Groq rate limits etc).
    """

    judges: list[Judge]
    threshold: float = 7.0
    aggregation: str = "median"
    max_parallel: int = 4

    def __post_init__(self) -> None:
        if self.aggregation not in AGGREGATORS:
            raise ValueError(
                f"Unknown aggregation {self.aggregation!r}; choose from "
                f"{list(AGGREGATORS)}"
            )
        if not self.judges:
            raise ValueError("EnsembleJudge needs at least one judge")

    @property
    def aggregator(self) -> Aggregator:
        return AGGREGATORS[self.aggregation]

    async def score(
        self, conversation: list[dict[str, str]]
    ) -> EnsembleVerdict:
        """Ask every judge in parallel; aggregate; return."""
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def _bounded(j: Judge) -> JudgeVerdict:
            async with semaphore:
                return await j.score(conversation)

        verdicts = await asyncio.gather(
            *(_bounded(j) for j in self.judges),
            return_exceptions=False,
        )
        valid_scores = [v.score for v in verdicts if v.error is None]
        aggregate = self.aggregator(valid_scores) if valid_scores else 0.0

        return EnsembleVerdict(
            score=aggregate,
            judges=list(verdicts),
            aggregation=self.aggregation,
        )

    def is_kept(self, verdict: EnsembleVerdict) -> bool:
        return verdict.score >= self.threshold

    def calibration(self) -> dict[str, dict]:
        """Per-judge calibration stats (mean, count)."""
        out: dict[str, dict] = {}
        for j in self.judges:
            if isinstance(j, ProviderJudge):
                out[j.name] = {
                    "mean": j.mean_score,
                    "count": len(j.scores_seen),
                }
        return out
