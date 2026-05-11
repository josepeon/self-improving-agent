"""T1.2: dashboard data layer — load + diff."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from self_improving_agent.dashboard.data import (
    DashboardState,
    diff_versions,
    load_dashboard_state,
)


@pytest.fixture
def populated(tmp_path):
    """Returns paths to a registry + reports dir with realistic content."""
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({
        "updated_at": "2026-02-27T19:20:01+00:00",
        "versions": [
            {
                "version": "v1.0",
                "base_model": "test-base",
                "adapter_path": "models/finetuned/v1.0",
                "created_at": "2026-02-25T15:07:07+00:00",
                "training_dataset_size": 2496,
                "status": "deprecated",
            },
            {
                "version": "v1.1",
                "base_model": "test-base",
                "adapter_path": "models/finetuned/v1.1",
                "created_at": "2026-02-26T11:27:52+00:00",
                "training_dataset_size": 2696,
                "status": "production",
            },
        ],
    }))

    reports = tmp_path / "reports"
    reports.mkdir()

    # Stagger mtimes so order is deterministic
    (reports / "v1.0_eval.json").write_text(json.dumps({
        "model_version": "v1.0",
        "timestamp": "2026-02-25T16:00:00+00:00",
        "pass_rate": 0.72,
        "passed_tests": 72,
        "total_tests": 100,
        "avg_latency_ms": 240.0,
        "category_scores": {"factual": 0.80, "tone": 0.65},
    }))
    time.sleep(0.01)
    (reports / "v1.1_eval.json").write_text(json.dumps({
        "model_version": "v1.1",
        "timestamp": "2026-02-26T12:00:00+00:00",
        "pass_rate": 0.85,
        "passed_tests": 85,
        "total_tests": 100,
        "avg_latency_ms": 220.0,
        "category_scores": {"factual": 0.88, "tone": 0.83, "safety": 0.91},
    }))
    return registry, reports


class TestLoadDashboardState:
    def test_empty_when_nothing_on_disk(self, tmp_path):
        state = load_dashboard_state(
            registry_path=tmp_path / "no_registry.json",
            reports_dir=tmp_path / "no_reports",
        )
        assert state.versions == []
        assert state.reports == []

    def test_loads_versions_and_reports(self, populated):
        registry, reports = populated
        state = load_dashboard_state(registry, reports)

        assert len(state.versions) == 2
        assert {v.version for v in state.versions} == {"v1.0", "v1.1"}
        assert len(state.reports) == 2

    def test_wires_reports_into_versions(self, populated):
        registry, reports = populated
        state = load_dashboard_state(registry, reports)
        by_version = {v.version: v for v in state.versions}
        assert len(by_version["v1.0"].reports) == 1
        assert len(by_version["v1.1"].reports) == 1
        assert by_version["v1.1"].latest_pass_rate == 0.85

    def test_production_version_helper(self, populated):
        registry, reports = populated
        state = load_dashboard_state(registry, reports)
        prod = state.production_version()
        assert prod is not None
        assert prod.version == "v1.1"

    def test_handles_corrupt_report(self, tmp_path):
        registry = tmp_path / "registry.json"
        registry.write_text(json.dumps({"versions": []}))
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "bad.json").write_text("not json")
        # Should not raise
        state = load_dashboard_state(registry, reports)
        assert state.reports == []

    def test_handles_corrupt_registry(self, tmp_path):
        registry = tmp_path / "registry.json"
        registry.write_text("not json")
        state = load_dashboard_state(registry, tmp_path / "reports")
        assert state.versions == []


class TestDiffVersions:
    def test_returns_delta_for_two_real_versions(self, populated):
        registry, reports = populated
        state = load_dashboard_state(registry, reports)
        diff = diff_versions(state, "v1.0", "v1.1")
        assert diff["pass_rate_a"] == 0.72
        assert diff["pass_rate_b"] == 0.85
        assert diff["pass_rate_delta"] == pytest.approx(0.13)

    def test_category_diff_handles_new_category(self, populated):
        registry, reports = populated
        state = load_dashboard_state(registry, reports)
        diff = diff_versions(state, "v1.0", "v1.1")
        # 'safety' exists only in v1.1
        assert "safety" in diff["categories"]
        assert diff["categories"]["safety"]["a"] is None
        assert diff["categories"]["safety"]["delta"] is None

    def test_unknown_version_returns_error(self, populated):
        registry, reports = populated
        state = load_dashboard_state(registry, reports)
        diff = diff_versions(state, "v1.0", "v9.9")
        assert "error" in diff
