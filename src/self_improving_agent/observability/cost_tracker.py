"""
Cost & Token Tracking

Records prompt + completion token counts and estimated USD cost for every
provider response. Aggregates per-model and per-script.

Design notes:

- A single thread-safe ``CostTracker`` is shared across the process. The
  providers don't construct it; they call ``record_usage(...)`` directly.
- Pricing is expressed in USD per million tokens. Unknown models cost $0
  and are surfaced in the summary so we never silently under-report.
- A run boundary (``begin_run()``) resets accumulators so scripts can
  report per-run totals separate from cumulative.
- Mirrors the design of ``multi_agent_planner/core/cost_tracker.py`` so
  the two projects can converge later.

Usage:

    from self_improving_agent.observability import (
        begin_run, record_usage, render_summary,
    )

    begin_run()
    # ... do work; providers call record_usage internally ...
    print(render_summary())
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterator


# ===========================================
# Pricing
# ===========================================


@dataclass(frozen=True)
class ModelPrice:
    """USD price per 1M tokens (input, output)."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_per_mtok
            + completion_tokens * self.output_per_mtok
        ) / 1_000_000


# Pricing snapshot. Free Groq tier is the default; OpenAI/Anthropic listed
# in case a new provider gets added. Mlx is local so $0.
_DEFAULT_PRICES: dict[tuple[str, str], ModelPrice] = {
    # Groq: free tier for SIA usage
    ("groq", "llama-3.3-70b-versatile"): ModelPrice(0.0, 0.0),
    ("groq", "llama-3.1-8b-instant"): ModelPrice(0.0, 0.0),
    ("groq", "gemma2-9b-it"): ModelPrice(0.0, 0.0),
    ("groq", "mixtral-8x7b-32768"): ModelPrice(0.0, 0.0),
    # MLX: local, always free
    ("mlx", "*"): ModelPrice(0.0, 0.0),
    # OpenAI (when configured)
    ("openai", "gpt-4o"): ModelPrice(2.5, 10.0),
    ("openai", "gpt-4o-mini"): ModelPrice(0.15, 0.6),
}


def get_price(provider: str, model: str) -> ModelPrice | None:
    if (provider, model) in _DEFAULT_PRICES:
        return _DEFAULT_PRICES[(provider, model)]
    if (provider, "*") in _DEFAULT_PRICES:
        return _DEFAULT_PRICES[(provider, "*")]
    return None


# ===========================================
# Data model
# ===========================================


@dataclass
class UsageRecord:
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    priced: bool  # False if we had no rate card


@dataclass
class UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    unpriced_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, record: UsageRecord) -> None:
        self.prompt_tokens += record.prompt_tokens
        self.completion_tokens += record.completion_tokens
        self.cost_usd += record.cost_usd
        self.calls += 1
        if not record.priced:
            self.unpriced_calls += 1

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "unpriced_calls": self.unpriced_calls,
        }


class BudgetExceededError(RuntimeError):
    """Raised when an in-progress run crosses its USD budget cap."""


# ===========================================
# Tracker
# ===========================================


class CostTracker:
    """Thread-safe accumulator. One per process, accessed via module helpers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[UsageRecord] = []
        self._budget_usd: float | None = None

    # ----- recording -----

    def record(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> UsageRecord:
        price = get_price(provider, model)
        if price is None:
            cost = 0.0
            priced = False
        else:
            cost = price.cost(prompt_tokens, completion_tokens)
            priced = True

        record = UsageRecord(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            priced=priced,
        )
        with self._lock:
            self._records.append(record)

        if self._budget_usd is not None and self.total().cost_usd > self._budget_usd:
            raise BudgetExceededError(
                f"Run exceeded budget: ${self.total().cost_usd:.4f} > "
                f"${self._budget_usd:.4f}"
            )

        return record

    # ----- queries -----

    def all_records(self) -> list[UsageRecord]:
        with self._lock:
            return list(self._records)

    def total(self) -> UsageTotals:
        totals = UsageTotals()
        for record in self.all_records():
            totals.add(record)
        return totals

    def by_model(self) -> dict[str, UsageTotals]:
        out: dict[str, UsageTotals] = {}
        for record in self.all_records():
            key = f"{record.provider}/{record.model}"
            out.setdefault(key, UsageTotals()).add(record)
        return out

    # ----- run lifecycle -----

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._budget_usd = None

    def set_budget(self, usd: float | None) -> None:
        self._budget_usd = usd


# ===========================================
# Module-level singleton + helpers
# ===========================================

_tracker = CostTracker()


def get_tracker() -> CostTracker:
    return _tracker


def record_usage(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> UsageRecord:
    return _tracker.record(provider, model, prompt_tokens, completion_tokens)


def begin_run(budget_usd: float | None = None) -> None:
    """Reset accumulators at the start of a script/pipeline run.

    Optionally set a hard budget cap; record_usage will raise
    BudgetExceededError after the threshold is crossed.
    """
    _tracker.reset()
    _tracker.set_budget(budget_usd)


# ===========================================
# Reporting
# ===========================================


def render_summary(tracker: CostTracker | None = None) -> str:
    """Human-readable summary suitable for terminal output."""
    tracker = tracker or _tracker
    totals = tracker.total()

    if totals.calls == 0:
        return "Cost: 0 calls (nothing to report)."

    lines = [
        f"Cost: ${totals.cost_usd:.4f}  "
        f"({totals.prompt_tokens:,} prompt + {totals.completion_tokens:,} completion "
        f"= {totals.total_tokens:,} tokens, {totals.calls} call(s))",
    ]

    by_model = tracker.by_model()
    if len(by_model) > 1:
        lines.append("  by model:")
        for key, t in sorted(by_model.items(), key=lambda kv: -kv[1].cost_usd):
            lines.append(
                f"    {key:<40} {t.calls:>3} call(s)  ${t.cost_usd:.4f}"
            )

    if totals.unpriced_calls:
        lines.append(
            f"  note: {totals.unpriced_calls} call(s) had no rate card "
            "and contributed $0 to the total."
        )

    return "\n".join(lines)


def to_dict(tracker: CostTracker | None = None) -> dict:
    """Structured snapshot for JSON logs."""
    tracker = tracker or _tracker
    return {
        "total": tracker.total().as_dict(),
        "by_model": {key: t.as_dict() for key, t in tracker.by_model().items()},
    }
