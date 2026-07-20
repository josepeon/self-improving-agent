"""
Feedback store

When a real user gets a bad response, that's the highest-value training
signal — but the pipeline previously had no path from "bad response in
prod" to "this goes in the next training batch."

This module persists feedback rows to JSONL and exposes helpers the
dataset builder uses to up-weight (or exclude) interactions based on
that signal.

Schema (one row per JSONL line):

    {
        "id": "fb_<sha1>",
        "interaction_id": "<original interaction id>",
        "rating": "good" | "bad",
        "correction": "<optional preferred response>",
        "user_message": "<prompt that produced the bad response>",
        "original_response": "<the response the user reacted to>",
        "model_version": "<which adapter served it>",
        "metadata": {...},
        "timestamp": <unix>
    }

Why a separate file from the interaction log: feedback is rare and
high-signal; we want to read it independently and apply different
retention policies (we keep feedback forever; rotated logs eventually
expire).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FeedbackRating(StrEnum):
    GOOD = "good"
    BAD = "bad"


@dataclass
class Feedback:
    """One operator-supplied feedback row."""

    interaction_id: str
    rating: FeedbackRating
    user_message: str
    original_response: str
    correction: str | None = None
    model_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def id(self) -> str:
        """Stable id derived from interaction_id + rating + correction.

        Using a deterministic id lets repeated submissions of the same
        feedback collapse into the same row instead of duplicating.
        """
        key = f"{self.interaction_id}|{self.rating}|{self.correction or ''}"
        return "fb_" + hashlib.sha1(key.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["id"] = self.id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Feedback:
        return cls(
            interaction_id=data["interaction_id"],
            rating=FeedbackRating(data["rating"]),
            user_message=data["user_message"],
            original_response=data["original_response"],
            correction=data.get("correction"),
            model_version=data.get("model_version"),
            metadata=data.get("metadata", {}),
            timestamp=float(data.get("timestamp", time.time())),
        )


# ---------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------


class FeedbackStore:
    """Append-only JSONL store with deterministic dedup on ``Feedback.id``."""

    def __init__(self, path: str | Path = "data/feedback/feedback.jsonl") -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        # id cache so add() doesn't re-scan the whole file per call
        # (it used to be O(file size) sync I/O inside async handlers).
        self._known_ids: set[str] | None = None

    def add(self, feedback: Feedback) -> bool:
        """Append the feedback. Returns False if an entry with the same id exists.

        Idempotent on (interaction_id, rating, correction) — repeated calls
        with the same triple are a no-op.
        """
        with self._lock:
            if self._known_ids is None:
                self._known_ids = self._existing_ids()
            if feedback.id in self._known_ids:
                logger.debug("Feedback %s already recorded; skipping", feedback.id)
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(feedback.to_dict()) + "\n")
            self._known_ids.add(feedback.id)
            return True

    def iter_all(self) -> Iterator[Feedback]:
        if not self.path.exists():
            return iter([])
        return self._iter_file()

    def _iter_file(self) -> Iterator[Feedback]:
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    yield Feedback.from_dict(data)
                except Exception as exc:
                    logger.warning("Skipping invalid feedback row: %s", exc)
                    continue

    def _existing_ids(self) -> set[str]:
        return {fb.id for fb in self.iter_all()}

    # ── Aggregations ───────────────────────────────────────────────────

    def counts_by_rating(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for fb in self.iter_all():
            out[str(fb.rating)] = out.get(str(fb.rating), 0) + 1
        return out

    def counts_by_version(self) -> dict[str, dict[str, int]]:
        """{version: {good: N, bad: N}} for the dashboard."""
        out: dict[str, dict[str, int]] = {}
        for fb in self.iter_all():
            version = fb.model_version or "<unknown>"
            bucket = out.setdefault(version, {"good": 0, "bad": 0})
            bucket[str(fb.rating)] = bucket.get(str(fb.rating), 0) + 1
        return out


# ---------------------------------------------------------------------
# Dataset builder integration
# ---------------------------------------------------------------------


def feedback_to_sft_pair(feedback: Feedback) -> dict | None:
    """Convert a Feedback row into an SFT pair if a correction is present.

    Only entries with a non-empty ``correction`` produce training data;
    'bad' ratings without corrections are signal that the original
    response was wrong but don't tell us what the right answer was.
    """
    if not feedback.correction or not feedback.correction.strip():
        return None

    return {
        "messages": [
            {"role": "user", "content": feedback.user_message},
            {"role": "assistant", "content": feedback.correction.strip()},
        ],
        "metadata": {
            "source": "feedback",
            "feedback_id": feedback.id,
            "rating": str(feedback.rating),
            "original_model_version": feedback.model_version,
            "timestamp": feedback.timestamp,
        },
    }


def bad_interaction_ids(store: FeedbackStore) -> set[str]:
    """Set of interaction ids the dataset builder should exclude.

    Anything rated 'bad' is dropped from default training data — the
    correction (if any) re-enters via ``feedback_to_sft_pair`` above.
    """
    return {
        fb.interaction_id for fb in store.iter_all() if fb.rating == FeedbackRating.BAD
    }
