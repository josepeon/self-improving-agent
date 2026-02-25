"""Self-improvement pipeline - orchestrates the full improvement loop.

Schedule (suggested):
    Hourly : aggregate interaction logs
    Daily  : evaluate current production model
    Weekly : build dataset, train new version, evaluate, promote
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from self_improving_agent.evaluation.harness import EvaluationHarness
from self_improving_agent.evaluation.models import EvaluationReport
from self_improving_agent.logging.storage import InteractionLogStore
from self_improving_agent.training.dataset_builder import DatasetBuilder
from self_improving_agent.training.finetune import FineTuner, LoRAConfig
from self_improving_agent.training.registry import (
    ModelRegistry,
    ModelStatus,
    ModelVersion,
)

logger = logging.getLogger(__name__)


# ── Protocols ─────────────────────────────────────────────────────────


class GenerateFn(Protocol):
    """Callable that matches the signature the evaluation harness expects."""

    async def __call__(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = ...,
        max_tokens: int = ...,
    ) -> str: ...


class ModelLoader(Protocol):
    """Load a generate function for a specific model version."""

    async def __call__(self, version: ModelVersion) -> GenerateFn: ...


# ── Configuration ─────────────────────────────────────────────────────


class PipelineConfig:
    """Knobs for the self-improvement pipeline."""

    def __init__(
        self,
        *,
        log_base_path: str = "logs",
        registry_path: str = "models/registry.json",
        reports_dir: str = "evaluation/reports",
        finetuned_dir: str = "models/finetuned",
        golden_tests_path: str | None = None,
        min_dataset_size: int = 100,
        min_quality_score: float = 0.8,
        regression_threshold: float = 0.95,
        lora_config: LoRAConfig | None = None,
    ) -> None:
        self.log_base_path = log_base_path
        self.registry_path = registry_path
        self.reports_dir = reports_dir
        self.finetuned_dir = finetuned_dir
        self.golden_tests_path = golden_tests_path
        self.min_dataset_size = min_dataset_size
        self.min_quality_score = min_quality_score
        self.regression_threshold = regression_threshold
        self.lora_config = lora_config or LoRAConfig()


# ── Pipeline ──────────────────────────────────────────────────────────


class SelfImprovementPipeline:
    """Orchestrate the complete self-improvement loop.

    The pipeline depends on a *model_loader* callback so it stays
    decoupled from any specific provider implementation.  The caller
    supplies a function that, given a :class:`ModelVersion`, returns
    an async generate function suitable for the evaluation harness.

    Parameters
    ----------
    model_loader:
        Async callable ``(ModelVersion) -> GenerateFn``.
    config:
        Pipeline configuration knobs.
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        config: PipelineConfig | None = None,
    ) -> None:
        self.cfg = config or PipelineConfig()
        self.model_loader = model_loader

        # Subsystems (lazy-initialised on first use)
        self._log_store: InteractionLogStore | None = None
        self._registry: ModelRegistry | None = None
        self._harness: EvaluationHarness | None = None
        self._finetuner: FineTuner | None = None
        self._dataset_builder: DatasetBuilder | None = None

    # ── Lazy accessors ────────────────────────────────────────────────

    @property
    def log_store(self) -> InteractionLogStore:
        if self._log_store is None:
            self._log_store = InteractionLogStore(self.cfg.log_base_path)
        return self._log_store

    @property
    def registry(self) -> ModelRegistry:
        if self._registry is None:
            self._registry = ModelRegistry(self.cfg.registry_path)
        return self._registry

    @property
    def harness(self) -> EvaluationHarness:
        if self._harness is None:
            self._harness = EvaluationHarness(self.cfg.golden_tests_path)
        return self._harness

    @property
    def finetuner(self) -> FineTuner:
        if self._finetuner is None:
            self._finetuner = FineTuner(
                output_dir=self.cfg.finetuned_dir,
                lora_config=self.cfg.lora_config,
            )
        return self._finetuner

    @property
    def dataset_builder(self) -> DatasetBuilder:
        if self._dataset_builder is None:
            self._dataset_builder = DatasetBuilder(self.log_store)
        return self._dataset_builder

    # ── Daily evaluation ──────────────────────────────────────────────

    async def run_daily_evaluation(self) -> EvaluationReport | None:
        """Evaluate the current production model.

        Returns the evaluation report, or ``None`` if no production
        model is registered.
        """
        prod = self.registry.get_production()
        if prod is None:
            logger.warning("No production model registered - skipping evaluation")
            return None

        logger.info("Running daily evaluation for %s", prod.version)
        generate_fn = await self.model_loader(prod)
        report = await self.harness.run_all(generate_fn, prod.version)

        # Persist report
        report_path = self._save_report(report, prod.version)
        prod.evaluation_report_path = str(report_path)
        prod.evaluation_pass_rate = report.pass_rate
        self.registry.register(prod)

        # Regression check
        self._check_regression(report, prod.version)

        logger.info(
            "Daily evaluation complete: %s - %.1f%% pass rate",
            prod.version,
            report.pass_rate * 100,
        )
        return report

    # ── Weekly training cycle ─────────────────────────────────────────

    async def run_weekly_training(
        self,
        *,
        lookback_days: int = 7,
    ) -> ModelVersion | None:
        """Build dataset, train, evaluate, and optionally promote.

        Returns the new :class:`ModelVersion` if training succeeded,
        or ``None`` if there was insufficient data.
        """
        end = datetime.now(UTC)
        start = end - timedelta(days=lookback_days)

        # 1. Build dataset
        dataset = await self.dataset_builder.build_sft_dataset(
            start,
            end,
            min_quality_score=self.cfg.min_quality_score,
        )
        if len(dataset) < self.cfg.min_dataset_size:
            logger.info(
                "Insufficient training data: %d examples (need %d)",
                len(dataset),
                self.cfg.min_dataset_size,
            )
            return None

        # 2. Train
        new_tag = f"v{end.strftime('%Y%m%d')}"
        logger.info("Training %s with %d examples", new_tag, len(dataset))

        adapter_path = await self.finetuner.train(dataset, new_tag)

        version = ModelVersion(
            version=new_tag,
            base_model=self.finetuner.base_model,
            adapter_path=adapter_path,
            training_dataset_size=len(dataset),
            status=ModelStatus.EVALUATING,
        )
        self.registry.register(version)

        # 3. Evaluate
        generate_fn = await self.model_loader(version)
        report = await self.harness.run_all(generate_fn, new_tag)
        report_path = self._save_report(report, new_tag)
        version.evaluation_report_path = str(report_path)
        version.evaluation_pass_rate = report.pass_rate
        self.registry.register(version)

        # 4. Promote or fail
        prod = self.registry.get_production()
        prod_report = self._load_latest_report(prod.version) if prod else None

        if prod_report is None or report.is_promotion_ready(prod_report):
            self.registry.promote_to_canary(new_tag)
            logger.info("Version %s promoted to canary", new_tag)
        else:
            self.registry.mark_failed(new_tag)
            logger.warning(
                "Version %s did not meet promotion criteria (%.1f%%)",
                new_tag,
                report.pass_rate * 100,
            )

        return version

    # ── Canary promotion ──────────────────────────────────────────────

    async def promote_canary_to_production(self) -> ModelVersion | None:
        """Promote the current canary to production if it still passes.

        Returns the promoted version, or ``None`` if no canary exists
        or evaluation failed.
        """
        canary = self.registry.get_canary()
        if canary is None:
            logger.info("No canary to promote")
            return None

        # Re-evaluate to be safe
        generate_fn = await self.model_loader(canary)
        report = await self.harness.run_all(generate_fn, canary.version)

        prod = self.registry.get_production()
        prod_report = self._load_latest_report(prod.version) if prod else None

        if prod_report is None or report.is_promotion_ready(prod_report):
            self.registry.promote_to_production(canary.version)
            logger.info("Canary %s promoted to production", canary.version)
            return canary

        logger.warning(
            "Canary %s failed re-evaluation - marking as failed",
            canary.version,
        )
        self.registry.mark_failed(canary.version)
        return None

    # ── Dataset statistics ────────────────────────────────────────────

    async def dataset_stats(self, *, lookback_days: int = 7) -> dict[str, Any]:
        """Return statistics about available training data."""
        end = datetime.now(UTC)
        start = end - timedelta(days=lookback_days)
        return await self.dataset_builder.dataset_stats(
            start,
            end,
            min_quality_score=self.cfg.min_quality_score,
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _save_report(self, report: EvaluationReport, version_tag: str) -> Path:
        """Persist an evaluation report to disk."""
        reports_dir = Path(self.cfg.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        path = reports_dir / f"{version_tag}_{date_str}.json"
        payload = {
            "model_version": report.model_version,
            "pass_rate": report.pass_rate,
            "total_tests": report.total_tests,
            "passed_tests": report.passed_tests,
            "failed_tests": report.failed_tests,
            "avg_latency_ms": report.avg_latency_ms,
            "category_scores": report.category_scores,
            "summary": report.summary(),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    def _load_latest_report(self, version_tag: str) -> EvaluationReport | None:
        """Load the most recent report for a version, if any."""
        reports_dir = Path(self.cfg.reports_dir)
        if not reports_dir.exists():
            return None
        candidates = sorted(reports_dir.glob(f"{version_tag}_*.json"), reverse=True)
        if not candidates:
            return None

        with open(candidates[0]) as f:
            data = json.load(f)

        # Reconstruct a minimal EvaluationReport from the summary
        return EvaluationReport(
            model_version=data.get("model_version", version_tag),
            results=[],
            pass_rate=data.get("pass_rate", 0.0),
        )

    def _check_regression(self, report: EvaluationReport, version_tag: str) -> None:
        """Log a warning if pass rate dropped significantly."""
        baseline = self._load_latest_report(version_tag)
        if baseline is None:
            return
        if (
            baseline.pass_rate > 0
            and report.pass_rate < baseline.pass_rate * self.cfg.regression_threshold
        ):
            logger.warning(
                "REGRESSION detected for %s: %.1f%% -> %.1f%%",
                version_tag,
                baseline.pass_rate * 100,
                report.pass_rate * 100,
            )
