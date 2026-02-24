"""Async JSONL storage for interaction logs.

Provides date-partitioned file storage with query and feedback support:
    logs/
      interactions/
        2026/02/24/
          interactions.jsonl
      feedback/
        feedback_pending.jsonl
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiofiles

from self_improving_agent.logging.interaction_log import InteractionLog


class InteractionLogStore:
    """Async JSONL store for interaction logs.

    Writes are append-only and protected by an asyncio lock so
    concurrent callers never interleave partial JSON lines.

    Storage hierarchy
    -----------------
    ``base_path/interactions/YYYY/MM/DD/interactions.jsonl``
        One JSONL file per calendar day (UTC).
    ``base_path/feedback/feedback_pending.jsonl``
        Feedback entries awaiting batch merge into interaction files.
    """

    def __init__(self, base_path: str = "logs") -> None:
        self.base_path = Path(base_path)
        self.interactions_path = self.base_path / "interactions"
        self.feedback_path = self.base_path / "feedback"
        self._write_lock = asyncio.Lock()

    # ── Writes ────────────────────────────────────────────────────────

    async def log_interaction(self, log: InteractionLog) -> Path:
        """Append *log* to the date-partitioned JSONL file.

        Returns the path of the file written to.
        """
        ts = datetime.fromisoformat(log.timestamp)
        date_path = self.interactions_path / ts.strftime("%Y/%m/%d")
        date_path.mkdir(parents=True, exist_ok=True)

        log_file = date_path / "interactions.jsonl"
        async with self._write_lock, aiofiles.open(log_file, "a") as f:
            await f.write(log.to_json() + "\n")
        return log_file

    # ── Reads / Queries ───────────────────────────────────────────────

    async def get_interactions(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        user_id: str | None = None,
        has_feedback: bool | None = None,
        failure_category: str | None = None,
    ) -> list[InteractionLog]:
        """Query interactions across a date range with optional filters.

        Parameters
        ----------
        start_date, end_date:
            Inclusive UTC date range.
        user_id:
            If set, only return logs for this user.
        has_feedback:
            ``True`` = only logs with feedback, ``False`` = only without.
        failure_category:
            If set, only return logs tagged with this failure category.
        """
        results: list[InteractionLog] = []
        current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = end_date.replace(hour=0, minute=0, second=0, microsecond=0)

        while current <= end:
            day_dir = self.interactions_path / current.strftime("%Y/%m/%d")
            log_file = day_dir / "interactions.jsonl"

            if log_file.exists():
                async with aiofiles.open(log_file) as f:
                    async for line in f:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        log = InteractionLog.from_json(stripped)
                        if self._matches(log, user_id, has_feedback, failure_category):
                            results.append(log)

            current += timedelta(days=1)

        return results

    async def get_interaction_by_id(
        self,
        interaction_id: str,
        date_hint: datetime | None = None,
    ) -> InteractionLog | None:
        """Look up a single interaction by its ID.

        Parameters
        ----------
        interaction_id:
            The UUID of the interaction.
        date_hint:
            If provided, only search the JSONL file for that day,
            otherwise scan all available days (slower).
        """
        if date_hint is not None:
            return await self._search_day(interaction_id, date_hint)

        # Full scan — walk every day directory
        if not self.interactions_path.exists():
            return None
        for year_dir in sorted(self.interactions_path.iterdir()):
            for month_dir in sorted(year_dir.iterdir()):
                for day_dir in sorted(month_dir.iterdir()):
                    day = datetime.strptime(
                        str(day_dir.relative_to(self.interactions_path)),
                        "%Y/%m/%d",
                    )
                    found = await self._search_day(interaction_id, day)
                    if found is not None:
                        return found
        return None

    # ── Feedback ──────────────────────────────────────────────────────

    async def add_feedback(
        self,
        interaction_id: str,
        feedback: str,
        feedback_text: str | None = None,
    ) -> None:
        """Queue feedback for an interaction.

        Feedback is appended to ``feedback_pending.jsonl``.
        Call :meth:`flush_feedback` to merge pending entries back into
        the main interaction files.
        """
        self.feedback_path.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {
            "interaction_id": interaction_id,
            "feedback": feedback,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if feedback_text is not None:
            entry["feedback_text"] = feedback_text

        pending = self.feedback_path / "feedback_pending.jsonl"
        async with self._write_lock, aiofiles.open(pending, "a") as f:
            await f.write(json.dumps(entry, separators=(",", ":")) + "\n")

    async def flush_feedback(self) -> int:
        """Merge pending feedback into interaction log files.

        Returns the number of feedback entries successfully merged.
        """
        pending = self.feedback_path / "feedback_pending.jsonl"
        if not pending.exists():
            return 0

        # Read all pending entries
        entries: list[dict[str, Any]] = []
        async with aiofiles.open(pending) as f:
            async for line in f:
                stripped = line.strip()
                if stripped:
                    entries.append(json.loads(stripped))

        merged = 0
        for entry in entries:
            iid = entry["interaction_id"]
            fb = entry["feedback"]
            fb_text = entry.get("feedback_text")

            # Try to find and update the log in-place
            if await self._update_interaction_feedback(iid, fb, fb_text):
                merged += 1

        # Truncate the pending file after merge
        if merged == len(entries):
            async with aiofiles.open(pending, "w") as f:
                await f.write("")
        else:
            # Re-write only un-merged entries
            merged_ids = set()
            remaining: list[str] = []
            for entry in entries:
                if entry["interaction_id"] in merged_ids:
                    continue
                # Check if this particular entry was merged
                log = await self.get_interaction_by_id(entry["interaction_id"])
                if log and log.user_feedback == entry["feedback"]:
                    merged_ids.add(entry["interaction_id"])
                else:
                    remaining.append(json.dumps(entry, separators=(",", ":")))
            async with aiofiles.open(pending, "w") as f:
                for line in remaining:
                    await f.write(line + "\n")

        return merged

    # ── Stats ─────────────────────────────────────────────────────────

    async def count_interactions(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> int:
        """Count interactions in a date range without loading them."""
        count = 0
        current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = end_date.replace(hour=0, minute=0, second=0, microsecond=0)

        while current <= end:
            day_dir = self.interactions_path / current.strftime("%Y/%m/%d")
            log_file = day_dir / "interactions.jsonl"

            if log_file.exists():
                async with aiofiles.open(log_file) as f:
                    async for line in f:
                        if line.strip():
                            count += 1
            current += timedelta(days=1)

        return count

    # ── Private helpers ───────────────────────────────────────────────

    @staticmethod
    def _matches(
        log: InteractionLog,
        user_id: str | None,
        has_feedback: bool | None,
        failure_category: str | None,
    ) -> bool:
        """Return True if *log* passes all provided filters."""
        if user_id is not None and log.user_id != user_id:
            return False
        if has_feedback is True and log.user_feedback is None:
            return False
        if has_feedback is False and log.user_feedback is not None:
            return False
        return not (
            failure_category is not None and log.failure_category != failure_category
        )

    async def _search_day(
        self,
        interaction_id: str,
        day: datetime,
    ) -> InteractionLog | None:
        """Search a single day's JSONL file for an interaction."""
        log_file = (
            self.interactions_path / day.strftime("%Y/%m/%d") / "interactions.jsonl"
        )
        if not log_file.exists():
            return None
        async with aiofiles.open(log_file) as f:
            async for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                data = json.loads(stripped)
                if data.get("interaction_id") == interaction_id:
                    return InteractionLog.from_dict(data)
        return None

    async def _update_interaction_feedback(
        self,
        interaction_id: str,
        feedback: str,
        feedback_text: str | None,
    ) -> bool:
        """Find an interaction across all day files and update its feedback.

        Rewrites the entire day file — fine for low-frequency feedback ops.
        Returns True if the interaction was found and updated.
        """
        if not self.interactions_path.exists():
            return False

        for year_dir in sorted(self.interactions_path.iterdir()):
            if not year_dir.is_dir():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                for day_dir in sorted(month_dir.iterdir()):
                    if not day_dir.is_dir():
                        continue
                    log_file = day_dir / "interactions.jsonl"
                    if not log_file.exists():
                        continue

                    lines: list[str] = []
                    found = False
                    async with aiofiles.open(log_file) as f:
                        async for line in f:
                            stripped = line.strip()
                            if not stripped:
                                continue
                            data = json.loads(stripped)
                            if data.get("interaction_id") == interaction_id:
                                data["user_feedback"] = feedback
                                if feedback_text is not None:
                                    data["user_feedback_text"] = feedback_text
                                found = True
                            lines.append(json.dumps(data, separators=(",", ":")))

                    if found:
                        async with self._write_lock, aiofiles.open(log_file, "w") as f:
                            for updated_line in lines:
                                await f.write(updated_line + "\n")
                        return True

        return False
