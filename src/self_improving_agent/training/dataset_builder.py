"""Build training datasets from interaction logs.

Filters high-quality interactions into SFT (supervised fine-tuning)
and DPO (direct preference optimization) formats suitable for
LoRA training with MLX.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from self_improving_agent.logging.interaction_log import InteractionLog
from self_improving_agent.logging.storage import InteractionLogStore


class DatasetBuilder:
    """Build training datasets from interaction logs.

    Filtering criteria for SFT:
    1. User feedback = "positive" OR quality_score >= threshold
    2. No failure categories (hallucination, refusal, etc.)
    3. Non-empty response text

    DPO pairs require matching positive+negative examples for the
    same type of prompt, or synthetic negative construction.
    """

    def __init__(self, log_store: InteractionLogStore) -> None:
        self.log_store = log_store

    # ── SFT dataset ───────────────────────────────────────────────────

    async def build_sft_dataset(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        min_quality_score: float = 0.8,
    ) -> list[dict[str, Any]]:
        """Build supervised fine-tuning dataset (Alpaca-style).

        Returns list of dicts with keys:
            instruction, input, output, history
        """
        logs = await self.log_store.get_interactions(
            start_date=start_date,
            end_date=end_date,
        )

        dataset: list[dict[str, Any]] = []
        for log in logs:
            if not self._is_sft_eligible(log, min_quality_score):
                continue
            dataset.append(self._to_alpaca(log))

        return dataset

    async def build_sft_and_save(
        self,
        start_date: datetime,
        end_date: datetime,
        output_path: str | Path,
        *,
        min_quality_score: float = 0.8,
    ) -> int:
        """Build SFT dataset and write to a JSONL file.

        Returns the number of examples written.
        """
        dataset = await self.build_sft_dataset(
            start_date,
            end_date,
            min_quality_score=min_quality_score,
        )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for item in dataset:
                f.write(json.dumps(item, separators=(",", ":")) + "\n")
        return len(dataset)

    # ── DPO dataset ───────────────────────────────────────────────────

    async def build_dpo_dataset(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        min_quality_score: float = 0.8,
    ) -> list[dict[str, Any]]:
        """Build DPO preference pairs.

        Pairs are constructed from interactions sharing similar prompts
        where one has positive feedback/high quality and another has
        negative feedback/low quality.

        Returns list of dicts with keys:
            prompt, chosen, rejected
        """
        logs = await self.log_store.get_interactions(
            start_date=start_date,
            end_date=end_date,
        )

        positives: list[InteractionLog] = []
        negatives: list[InteractionLog] = []

        for log in logs:
            if not log.response_text:
                continue
            # Explicit negative signals win over a high automated quality
            # score — a user-rejected response must never land on the
            # "chosen" side of a DPO pair.
            if log.user_feedback == "negative" or log.failure_category:
                negatives.append(log)
            elif log.user_feedback == "positive" or (
                log.quality_score is not None and log.quality_score >= min_quality_score
            ):
                positives.append(log)

        # Pair by matching user_message (exact match for now)
        negative_map: dict[str, InteractionLog] = {}
        for neg in negatives:
            negative_map.setdefault(neg.user_message, neg)

        pairs: list[dict[str, Any]] = []
        for pos in positives:
            matched_neg = negative_map.get(pos.user_message)
            if matched_neg is None:
                continue
            pairs.append(
                {
                    "prompt": pos.user_message,
                    "chosen": pos.response_text,
                    "rejected": matched_neg.response_text,
                }
            )

        return pairs

    # ── Statistics ────────────────────────────────────────────────────

    async def dataset_stats(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        min_quality_score: float = 0.8,
    ) -> dict[str, Any]:
        """Return statistics about available training data."""
        logs = await self.log_store.get_interactions(
            start_date=start_date,
            end_date=end_date,
        )
        total = len(logs)
        eligible = sum(
            1 for log in logs if self._is_sft_eligible(log, min_quality_score)
        )
        with_feedback = sum(1 for log in logs if log.user_feedback is not None)
        with_failures = sum(1 for log in logs if log.failure_category is not None)

        return {
            "total_interactions": total,
            "sft_eligible": eligible,
            "with_feedback": with_feedback,
            "with_failures": with_failures,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
        }

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _is_sft_eligible(log: InteractionLog, min_quality: float) -> bool:
        """Check if a log entry qualifies for SFT training."""
        # Must have a response
        if not log.response_text:
            return False
        # Must have a user message
        if not log.user_message:
            return False
        # No failures
        if log.failure_category:
            return False
        # Explicit negative feedback disqualifies
        if log.user_feedback == "negative":
            return False
        # Must have positive signal: explicit feedback or quality score
        if log.user_feedback == "positive":
            return True
        return log.quality_score is not None and log.quality_score >= min_quality

    @staticmethod
    def _to_alpaca(log: InteractionLog) -> dict[str, Any]:
        """Convert an interaction log to Alpaca-style training format."""
        return {
            "instruction": log.system_prompt,
            "input": log.user_message,
            "output": log.response_text,
            "history": log.conversation_history,
        }
