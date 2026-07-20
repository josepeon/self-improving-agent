"""Tests for SelfImprovementPipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from self_improving_agent.evaluation.models import EvaluationReport
from self_improving_agent.pipeline.self_improve import (
    PipelineConfig,
    SelfImprovementPipeline,
)
from self_improving_agent.training.registry import ModelStatus, ModelVersion

# ── Helpers ───────────────────────────────────────────────────────────


def _dummy_generate_fn() -> AsyncMock:
    return AsyncMock(return_value="Hello from model")


def _dummy_model_loader() -> AsyncMock:
    return AsyncMock(return_value=_dummy_generate_fn())


def _make_report(
    version: str = "v1.0",
    pass_rate: float = 0.9,
    **kw: object,
) -> EvaluationReport:
    return EvaluationReport(
        model_version=version,
        pass_rate=pass_rate,
        total_tests=10,
        passed_tests=int(10 * pass_rate),
        failed_tests=int(10 * (1 - pass_rate)),
        **kw,  # type: ignore[arg-type]
    )


def _make_pipeline(
    tmp_path: Path,
    *,
    model_loader: AsyncMock | None = None,
    harness_report: EvaluationReport | None = None,
) -> SelfImprovementPipeline:
    cfg = PipelineConfig(
        log_base_path=str(tmp_path / "logs"),
        registry_path=str(tmp_path / "registry.json"),
        reports_dir=str(tmp_path / "reports"),
        finetuned_dir=str(tmp_path / "finetuned"),
        min_dataset_size=2,
    )
    pipe = SelfImprovementPipeline(
        model_loader=model_loader or _dummy_model_loader(),
        config=cfg,
    )
    if harness_report is not None:
        mock_harness = AsyncMock()
        mock_harness.run_all = AsyncMock(return_value=harness_report)
        pipe._harness = mock_harness
    return pipe


# ── PipelineConfig ────────────────────────────────────────────────────


class TestPipelineConfig:
    def test_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.min_dataset_size == 100
        assert cfg.min_quality_score == 0.8
        assert cfg.regression_threshold == 0.95

    def test_custom(self) -> None:
        cfg = PipelineConfig(min_dataset_size=50, regression_threshold=0.9)
        assert cfg.min_dataset_size == 50
        assert cfg.regression_threshold == 0.9


# ── Lazy accessors ────────────────────────────────────────────────────


class TestLazyAccessors:
    def test_log_store_created_lazily(self, tmp_path: Path) -> None:
        pipe = _make_pipeline(tmp_path)
        assert pipe._log_store is None
        _ = pipe.log_store
        assert pipe._log_store is not None

    def test_registry_created_lazily(self, tmp_path: Path) -> None:
        pipe = _make_pipeline(tmp_path)
        assert pipe._registry is None
        _ = pipe.registry
        assert pipe._registry is not None

    def test_finetuner_created_lazily(self, tmp_path: Path) -> None:
        pipe = _make_pipeline(tmp_path)
        assert pipe._finetuner is None
        _ = pipe.finetuner
        assert pipe._finetuner is not None

    def test_dataset_builder_created_lazily(self, tmp_path: Path) -> None:
        pipe = _make_pipeline(tmp_path)
        assert pipe._dataset_builder is None
        _ = pipe.dataset_builder
        assert pipe._dataset_builder is not None


# ── Daily evaluation ──────────────────────────────────────────────────


class TestDailyEvaluation:
    async def test_no_production_returns_none(self, tmp_path: Path) -> None:
        pipe = _make_pipeline(tmp_path)
        result = await pipe.run_daily_evaluation()
        assert result is None

    async def test_evaluates_production(self, tmp_path: Path) -> None:
        report = _make_report("v1.0", pass_rate=0.9)
        pipe = _make_pipeline(tmp_path, harness_report=report)
        v = ModelVersion(
            version="v1.0",
            base_model="test",
            status=ModelStatus.PRODUCTION,
        )
        pipe.registry.register(v)
        result = await pipe.run_daily_evaluation()

        assert result is not None
        assert result.pass_rate == 0.9
        reports = list(Path(pipe.cfg.reports_dir).glob("v1.0_*.json"))
        assert len(reports) == 1

    async def test_saves_report_json(self, tmp_path: Path) -> None:
        report = _make_report("v1.0")
        pipe = _make_pipeline(tmp_path, harness_report=report)
        v = ModelVersion(
            version="v1.0", base_model="test", status=ModelStatus.PRODUCTION
        )
        pipe.registry.register(v)
        await pipe.run_daily_evaluation()

        reports = list(Path(pipe.cfg.reports_dir).glob("v1.0_*.json"))
        data = json.loads(reports[0].read_text())
        assert data["model_version"] == "v1.0"
        assert "pass_rate" in data

    async def test_updates_registry_after_eval(self, tmp_path: Path) -> None:
        report = _make_report("v1.0", pass_rate=0.85)
        pipe = _make_pipeline(tmp_path, harness_report=report)
        v = ModelVersion(
            version="v1.0", base_model="test", status=ModelStatus.PRODUCTION
        )
        pipe.registry.register(v)
        await pipe.run_daily_evaluation()

        updated = pipe.registry.get("v1.0")
        assert updated.evaluation_pass_rate == 0.85


# ── Weekly training ───────────────────────────────────────────────────


class TestWeeklyTraining:
    async def test_insufficient_data_returns_none(self, tmp_path: Path) -> None:
        pipe = _make_pipeline(tmp_path)
        with patch.object(
            pipe.dataset_builder,
            "build_sft_dataset",
            new_callable=AsyncMock,
            return_value=[],  # empty dataset
        ):
            result = await pipe.run_weekly_training()
        assert result is None

    async def test_trains_and_promotes_to_canary(self, tmp_path: Path) -> None:
        report = _make_report("new", pass_rate=0.95)
        pipe = _make_pipeline(tmp_path, harness_report=report)
        dataset = [
            {"instruction": "", "input": f"Q{i}", "output": f"A{i}", "history": []}
            for i in range(5)
        ]

        with (
            patch.object(
                pipe.dataset_builder,
                "build_sft_dataset",
                new_callable=AsyncMock,
                return_value=dataset,
            ),
            patch.object(
                pipe.finetuner,
                "train",
                new_callable=AsyncMock,
                return_value=str(tmp_path / "adapters"),
            ),
        ):
            result = await pipe.run_weekly_training()

        assert result is not None
        v = pipe.registry.get(result.version)
        assert v.status == ModelStatus.CANARY

    async def test_fails_when_below_baseline(self, tmp_path: Path) -> None:
        bad_report = _make_report("new", pass_rate=0.5)
        pipe = _make_pipeline(tmp_path, harness_report=bad_report)

        # Register existing production with high pass rate
        prod = ModelVersion(
            version="v1.0", base_model="test", status=ModelStatus.PRODUCTION
        )
        pipe.registry.register(prod)
        pipe._save_report(_make_report("v1.0", pass_rate=0.95), "v1.0")

        dataset = [
            {"instruction": "", "input": f"Q{i}", "output": f"A{i}", "history": []}
            for i in range(5)
        ]

        with (
            patch.object(
                pipe.dataset_builder,
                "build_sft_dataset",
                new_callable=AsyncMock,
                return_value=dataset,
            ),
            patch.object(
                pipe.finetuner,
                "train",
                new_callable=AsyncMock,
                return_value=str(tmp_path / "adapters"),
            ),
        ):
            result = await pipe.run_weekly_training()

        assert result is not None
        v = pipe.registry.get(result.version)
        assert v.status == ModelStatus.FAILED


# ── Canary promotion ──────────────────────────────────────────────────


class TestCanaryPromotion:
    async def test_no_canary_returns_none(self, tmp_path: Path) -> None:
        pipe = _make_pipeline(tmp_path)
        result = await pipe.promote_canary_to_production()
        assert result is None

    async def test_promotes_passing_canary(self, tmp_path: Path) -> None:
        report = _make_report("v2.0", pass_rate=0.95)
        pipe = _make_pipeline(tmp_path, harness_report=report)
        canary = ModelVersion(
            version="v2.0", base_model="test", status=ModelStatus.CANARY
        )
        pipe.registry.register(canary)
        result = await pipe.promote_canary_to_production()

        assert result is not None
        assert pipe.registry.get("v2.0").status == ModelStatus.PRODUCTION

    async def test_fails_bad_canary(self, tmp_path: Path) -> None:
        bad_report = _make_report("v2.0", pass_rate=0.3)
        pipe = _make_pipeline(tmp_path, harness_report=bad_report)

        prod = ModelVersion(
            version="v1.0", base_model="test", status=ModelStatus.PRODUCTION
        )
        pipe.registry.register(prod)
        pipe._save_report(_make_report("v1.0", pass_rate=0.95), "v1.0")

        canary = ModelVersion(
            version="v2.0", base_model="test", status=ModelStatus.CANARY
        )
        pipe.registry.register(canary)
        result = await pipe.promote_canary_to_production()

        assert result is None
        assert pipe.registry.get("v2.0").status == ModelStatus.FAILED


# ── Dataset stats ─────────────────────────────────────────────────────


class TestDatasetStats:
    async def test_returns_stats(self, tmp_path: Path) -> None:
        pipe = _make_pipeline(tmp_path)
        with patch.object(
            pipe.dataset_builder,
            "dataset_stats",
            new_callable=AsyncMock,
            return_value={"total_interactions": 42, "sft_eligible": 10},
        ):
            stats = await pipe.dataset_stats(lookback_days=7)
        assert stats["total_interactions"] == 42


# ── Report helpers ────────────────────────────────────────────────────


class TestReportHelpers:
    def test_save_and_load_report(self, tmp_path: Path) -> None:
        pipe = _make_pipeline(tmp_path)
        report = _make_report("v1.0", pass_rate=0.85)
        pipe._save_report(report, "v1.0")

        loaded = pipe._load_latest_report("v1.0")
        assert loaded is not None
        assert loaded.pass_rate == 0.85

    def test_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        pipe = _make_pipeline(tmp_path)
        assert pipe._load_latest_report("v99.0") is None

    def test_regression_check_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pipe = _make_pipeline(tmp_path)
        # Save a good baseline, then reload it the way the pipeline does
        pipe._save_report(_make_report("v1.0", pass_rate=0.95), "v1.0")
        baseline = pipe._load_latest_report("v1.0")
        # Check with a regressed report
        regressed = _make_report("v1.0", pass_rate=0.5)
        with caplog.at_level("WARNING"):
            pipe._check_regression(regressed, "v1.0", baseline)
        assert "REGRESSION" in caplog.text
