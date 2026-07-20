"""
Knowledge distillation: teacher (cloud) -> student (local).

Groq 70B answers questions well. SIA's local 8B is the deployment
target. There's no single command that takes a question pool, fetches
the teacher's responses, filters by quality, and produces SFT pairs
the existing training pipeline can consume.

This module is that command. Three pieces:

  - QuestionPool — iterable of user prompts. Pluggable; defaults to
    reading a JSONL of {"prompt": "..."} rows.
  - DistillationRunner — coordinates teacher calls, filters via a Judge
    (T1.3 EnsembleJudge by default), and writes SFT JSONL the existing
    dataset builder can ingest.
  - run_distillation() — convenience entry point. Used by
    scripts/distill_from_groq.py.

The pipeline doesn't bypass quality filtering by default — every teacher
response is judged before becoming training data. The bar is configurable
via the judge's threshold.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from self_improving_agent.providers.base import (
    BaseModelProvider,
    GenerationConfig,
)
from self_improving_agent.training.judges import (
    EnsembleJudge,
    Judge,
    ProviderJudge,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Question pool
# ---------------------------------------------------------------------


class QuestionPool(Protocol):
    """Anything that yields prompts as strings."""

    def __iter__(self) -> Iterator[str]: ...


@dataclass
class JsonlQuestionPool:
    """Reads {"prompt": "..."} (or {"question": ...}) rows from a JSONL file."""

    path: Path
    field_names: tuple[str, ...] = ("prompt", "question", "user")

    def __iter__(self) -> Iterator[str]:
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for field in self.field_names:
                    val = row.get(field)
                    if isinstance(val, str) and val.strip():
                        yield val.strip()
                        break


@dataclass
class ListQuestionPool:
    """In-memory list of prompts — useful for testing and quick runs."""

    prompts: list[str]

    def __iter__(self) -> Iterator[str]:
        return iter(self.prompts)


# ---------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------


@dataclass
class DistilledPair:
    """One filter-passing SFT pair ready for training."""

    prompt: str
    response: str
    judge_score: float
    teacher_model: str
    teacher_provider: str

    def to_sft_row(self) -> dict:
        """Serialize as the row shape dataset_builder consumes."""
        return {
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.response},
            ],
            "metadata": {
                "source": "distillation",
                "teacher_provider": self.teacher_provider,
                "teacher_model": self.teacher_model,
                "judge_score": self.judge_score,
            },
        }


@dataclass
class DistillationReport:
    """End-of-run summary."""

    questions_seen: int = 0
    teacher_errors: int = 0
    judge_errors: int = 0
    pairs_kept: int = 0
    pairs_dropped: int = 0
    output_path: str | None = None
    judge_calibration: dict[str, dict] = field(default_factory=dict)

    @property
    def keep_rate(self) -> float:
        total = self.pairs_kept + self.pairs_dropped
        return self.pairs_kept / total if total else 0.0

    def render(self) -> str:
        lines = [
            "Distillation report:",
            f"  questions seen:     {self.questions_seen}",
            f"  teacher errors:     {self.teacher_errors}",
            f"  judge errors:       {self.judge_errors}",
            f"  pairs kept:         {self.pairs_kept}",
            f"  pairs dropped:      {self.pairs_dropped}",
            f"  keep rate:          {self.keep_rate:.0%}",
        ]
        if self.output_path:
            lines.append(f"  wrote:              {self.output_path}")
        if self.judge_calibration:
            lines.append("  judge calibration:")
            for name, stats in self.judge_calibration.items():
                mean = stats.get("mean")
                count = stats.get("count")
                mean_str = f"{mean:.2f}" if mean is not None else "—"
                lines.append(f"    {name:<24} mean={mean_str}  count={count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------


@dataclass
class DistillationRunner:
    """Drive a full teacher → judge → SFT-pair pipeline."""

    teacher: BaseModelProvider
    judge: EnsembleJudge
    output_path: Path
    teacher_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(temperature=0.3, max_tokens=512)
    )
    max_concurrent: int = 4
    system_prompt: str | None = None

    async def run(self, pool: QuestionPool) -> DistillationReport:
        report = DistillationReport()
        report.output_path = str(self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _bounded(question: str) -> DistilledPair | None:
            async with semaphore:
                return await self._distill_one(question, report)

        questions = list(pool)
        report.questions_seen = len(questions)

        # Up to max_concurrent teacher+judge chains in flight at once;
        # results are written in input order once all complete.
        pairs = await asyncio.gather(*(_bounded(q) for q in questions))

        with open(self.output_path, "w") as out:
            for pair in pairs:
                if pair is None:
                    report.pairs_dropped += 1
                    continue
                out.write(json.dumps(pair.to_sft_row()) + "\n")
                report.pairs_kept += 1

        report.judge_calibration = self.judge.calibration()
        return report

    async def _distill_one(
        self, question: str, report: DistillationReport
    ) -> DistilledPair | None:
        # 1) Teacher generates
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": question})

        try:
            response = await self.teacher.generate(messages, self.teacher_config)
        except Exception as exc:
            logger.warning("Teacher call failed for %r: %s", question[:60], exc)
            report.teacher_errors += 1
            return None

        # 2) Judge filters
        conversation = [
            *messages,
            {"role": "assistant", "content": response.text},
        ]
        try:
            verdict = await self.judge.score(conversation)
        except Exception as exc:
            logger.warning("Judge ensemble failed for %r: %s", question[:60], exc)
            report.judge_errors += 1
            return None

        if not self.judge.is_kept(verdict):
            return None

        return DistilledPair(
            prompt=question,
            response=response.text,
            judge_score=verdict.score,
            teacher_model=response.model,
            teacher_provider=response.provider,
        )


# ---------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------


async def run_distillation(
    pool: QuestionPool,
    output_path: str | Path,
    *,
    teacher: BaseModelProvider | None = None,
    judges: list[Judge] | None = None,
    judge_threshold: float = 7.0,
    system_prompt: str | None = None,
    max_concurrent: int = 4,
) -> DistillationReport:
    """High-level helper. Default teacher = GroqProvider; default judge = same model."""
    from self_improving_agent.providers.groq_provider import GroqProvider

    teacher = teacher or GroqProvider()
    if not judges:
        judges = [ProviderJudge(GroqProvider(), name="groq")]

    judge = EnsembleJudge(judges=judges, threshold=judge_threshold)
    runner = DistillationRunner(
        teacher=teacher,
        judge=judge,
        output_path=Path(output_path),
        max_concurrent=max_concurrent,
        system_prompt=system_prompt,
    )
    return await runner.run(pool)
