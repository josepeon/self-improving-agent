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

import contextvars
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

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


def register_price(provider: str, model: str, price: ModelPrice) -> None:
    """Add a (provider, model) → price entry to the global table.

    Downstream packages (e.g. multi_agent_planner) call this on import to
    register provider/model combinations SIA itself doesn't ship pricing
    for. Use ``model="*"`` for a provider-wide fallback.
    """
    _DEFAULT_PRICES[(provider, model)] = price


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
    # Attribution: which subsystem made this call. Defaults to whatever
    # the active ``attribute()`` context is, or ``"unknown"``. Used by
    # multi_agent_planner to break down costs per agent role; used by
    # SIA itself to separate eval / distillation / serving traffic.
    agent_role: str = "unknown"


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
        agent_role: str | None = None,
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
            agent_role=agent_role or _current_role.get(),
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

    def by_agent(self) -> dict[str, UsageTotals]:
        """{agent_role: totals} — breakdown by who issued the call."""
        out: dict[str, UsageTotals] = {}
        for record in self.all_records():
            out.setdefault(record.agent_role, UsageTotals()).add(record)
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
_current_role: contextvars.ContextVar[str] = contextvars.ContextVar(
    "agent_role", default="unknown"
)


def get_tracker() -> CostTracker:
    return _tracker


def record_usage(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    agent_role: str | None = None,
) -> UsageRecord:
    return _tracker.record(
        provider, model, prompt_tokens, completion_tokens, agent_role
    )


@contextmanager
def attribute(role: str) -> Iterator[None]:
    """Scope subsequent ``record_usage`` calls to ``role``.

    Nested contexts honor the innermost role. Use this to break down
    cost by subsystem:

        with attribute("planner"):
            ... LLM calls here are attributed to "planner" ...
    """
    token = _current_role.set(role)
    try:
        yield
    finally:
        _current_role.reset(token)


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

    by_agent = tracker.by_agent()
    # Only render the agent block when attribution was actually used —
    # all-"unknown" means nobody scoped their calls and the line would
    # just duplicate the total.
    if by_agent and not (len(by_agent) == 1 and "unknown" in by_agent):
        lines.append("  by agent:")
        for role, t in sorted(by_agent.items(), key=lambda kv: -kv[1].cost_usd):
            lines.append(
                f"    {role:<16} {t.calls:>3} call(s)  "
                f"{t.total_tokens:>7,} tok  ${t.cost_usd:.4f}"
            )

    by_model = tracker.by_model()
    if len(by_model) > 1:
        lines.append("  by model:")
        for key, t in sorted(by_model.items(), key=lambda kv: -kv[1].cost_usd):
            lines.append(f"    {key:<40} {t.calls:>3} call(s)  ${t.cost_usd:.4f}")

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
        "by_agent": {role: t.as_dict() for role, t in tracker.by_agent().items()},
        "by_model": {key: t.as_dict() for key, t in tracker.by_model().items()},
    }
