"""Tests for run_improvement_cycle.py — version management and data merging."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_improvement_cycle import get_next_version


# ── get_next_version ─────────────────────────────────────────────────


class TestGetNextVersion:
    """Tests for automatic version tag generation."""

    def test_no_current_production(self) -> None:
        result = get_next_version(None)
        assert result == "v1.0-oh"

    def test_increment_minor(self) -> None:
        current = {"version": "v1.2-oh", "status": "production"}
        result = get_next_version(current)
        assert result == "v1.3-oh"

    def test_increment_from_v1_0(self) -> None:
        current = {"version": "v1.0-oh", "status": "production"}
        result = get_next_version(current)
        assert result == "v1.1-oh"

    def test_increment_from_v2_5(self) -> None:
        current = {"version": "v2.5-oh", "status": "production"}
        result = get_next_version(current)
        assert result == "v2.6-oh"

    def test_handles_missing_version_key(self) -> None:
        current = {"status": "production"}
        # Falls back to default parsing of "v1.0-oh"
        result = get_next_version(current)
        assert result == "v1.1-oh"


# ── ModelVersion lineage ─────────────────────────────────────────────


class TestModelVersionLineage:
    """Tests for data_sources and parent_version fields."""

    def test_data_sources_default_empty(self) -> None:
        from self_improving_agent.training.registry import ModelVersion

        v = ModelVersion(version="v1.0", base_model="test")
        assert v.data_sources == {}
        assert v.parent_version is None

    def test_data_sources_round_trip(self) -> None:
        from self_improving_agent.training.registry import ModelVersion

        sources = {"sft": 2496, "self_play_v1": 200, "real": 6}
        v = ModelVersion(
            version="v1.3-oh",
            base_model="test",
            data_sources=sources,
            parent_version="v1.2-oh",
        )
        d = v.to_dict()
        restored = ModelVersion.from_dict(d)
        assert restored.data_sources == sources
        assert restored.parent_version == "v1.2-oh"

    def test_lineage_in_registry(self, tmp_path: Path) -> None:
        from self_improving_agent.training.registry import (
            ModelRegistry,
            ModelStatus,
            ModelVersion,
        )

        reg = ModelRegistry(tmp_path / "registry.json")
        v = ModelVersion(
            version="v1.3-oh",
            base_model="test",
            status=ModelStatus.PRODUCTION,
            data_sources={"sft": 2496, "real": 10},
            parent_version="v1.2-oh",
        )
        reg.register(v)

        # Reload from disk
        reg2 = ModelRegistry(tmp_path / "registry.json")
        loaded = reg2.get("v1.3-oh")
        assert loaded is not None
        assert loaded.data_sources == {"sft": 2496, "real": 10}
        assert loaded.parent_version == "v1.2-oh"
