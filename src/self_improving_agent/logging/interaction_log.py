"""Interaction log schema — complete record of every LLM interaction.

Every request/response cycle is captured as an InteractionLog for:
- Evaluation (golden test comparison)
- Training data extraction (SFT / DPO dataset building)
- Quality monitoring (latency, fallback rate, feedback)
- Debugging (full replay of any interaction)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class InteractionLog:
    """Complete record of a single LLM interaction.

    Fields are grouped by lifecycle stage:
    1. Identity — who/when
    2. Input — what was asked
    3. Model — which model answered
    4. Output — what was returned
    5. Performance — how fast
    6. Evaluation — how good (filled later)
    """

    # ── Identity ──────────────────────────────────────────────────────
    interaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Request context
    user_id: str = ""
    session_id: str = ""
    request_id: str = ""

    # ── Input ─────────────────────────────────────────────────────────
    user_message: str = ""
    system_prompt: str = ""
    system_prompt_version: str = ""
    conversation_history: list[dict[str, str]] = field(default_factory=list)

    # Memory / RAG context
    memory_context: dict[str, Any] = field(default_factory=dict)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)

    # ── Model ─────────────────────────────────────────────────────────
    provider: str = ""  # "groq", "mlx"
    model_name: str = ""
    model_version: str = ""  # "base", "v1.0", "v1.1" for fine-tuned
    is_fallback: bool = False
    quality_mode: str = "default"  # "fast", "default", "quality"

    # Generation config
    temperature: float = 0.7
    max_tokens: int = 150

    # Tool usage (future)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)

    # ── Output ────────────────────────────────────────────────────────
    response_text: str = ""
    response_tokens: int = 0

    # ── Performance ───────────────────────────────────────────────────
    latency_ms: float = 0.0
    time_to_first_token_ms: float | None = None

    # Cost (estimated)
    input_tokens: int = 0
    cost_usd: float = 0.0

    # ── Evaluation (filled later) ─────────────────────────────────────
    user_feedback: str | None = None  # "positive", "negative"
    user_feedback_text: str | None = None
    failure_category: str | None = None  # "hallucination", "refusal", "off_topic"
    quality_score: float | None = None  # 0.0-1.0 from evaluation harness

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary (JSON-safe)."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to a single-line JSON string (for JSONL storage)."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InteractionLog:
        """Create from a dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, json_str: str) -> InteractionLog:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_provider_response(
        cls,
        response: Any,
        *,
        user_message: str = "",
        system_prompt: str = "",
        quality_mode: str = "default",
        **kwargs: Any,
    ) -> InteractionLog:
        """Create a log entry from a ProviderResponse.

        Convenience factory that maps ProviderResponse fields
        to the corresponding log fields.
        """
        return cls(
            user_message=user_message,
            system_prompt=system_prompt,
            quality_mode=quality_mode,
            provider=getattr(response, "provider", ""),
            model_name=getattr(response, "model", ""),
            is_fallback=getattr(response, "is_fallback", False),
            response_text=getattr(response, "text", ""),
            response_tokens=getattr(response, "tokens_used", 0),
            latency_ms=getattr(response, "latency_ms", 0.0),
            **kwargs,
        )
