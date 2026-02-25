"""Tests for ModelRegistry and ModelVersion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_improving_agent.training.registry import (
    ModelRegistry,
    ModelStatus,
    ModelVersion,
)


def _make_version(version: str = "v1.0", **kw: object) -> ModelVersion:
    defaults = {
        "version": version,
        "base_model": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "status": ModelStatus.TRAINING,
    }
    defaults.update(kw)
    return ModelVersion(**defaults)  # type: ignore[arg-type]


# ── ModelVersion ──────────────────────────────────────────────────────


class TestModelVersion:
    def test_defaults(self) -> None:
        v = ModelVersion(version="v1.0", base_model="test-model")
        assert v.status == ModelStatus.TRAINING
        assert v.adapter_path is None
        assert v.training_dataset_size == 0

    def test_round_trip_dict(self) -> None:
        v = _make_version(adapter_path="/adapters/v1.0")
        d = v.to_dict()
        restored = ModelVersion.from_dict(d)
        assert restored.version == v.version
        assert restored.adapter_path == "/adapters/v1.0"

    def test_from_dict_ignores_unknown_keys(self) -> None:
        d = {"version": "v1.0", "base_model": "m", "unknown_key": 42}
        v = ModelVersion.from_dict(d)
        assert v.version == "v1.0"


# ── ModelStatus enum ──────────────────────────────────────────────────


class TestModelStatus:
    def test_values(self) -> None:
        assert ModelStatus.TRAINING == "training"
        assert ModelStatus.PRODUCTION == "production"
        assert ModelStatus.CANARY == "canary"
        assert ModelStatus.DEPRECATED == "deprecated"
        assert ModelStatus.FAILED == "failed"

    def test_is_string(self) -> None:
        assert isinstance(ModelStatus.TRAINING, str)


# ── ModelRegistry ─────────────────────────────────────────────────────


class TestRegistryBasics:
    def test_register_and_get(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        v = _make_version()
        reg.register(v)
        assert reg.get("v1.0") is not None
        assert reg.get("v1.0").base_model == v.base_model

    def test_get_missing(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        assert reg.get("nonexistent") is None

    def test_list_versions(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(_make_version("v1.0"))
        reg.register(_make_version("v2.0"))
        assert len(reg.list_versions()) == 2

    def test_list_versions_filtered(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(_make_version("v1.0", status=ModelStatus.PRODUCTION))
        reg.register(_make_version("v2.0", status=ModelStatus.TRAINING))
        prods = reg.list_versions(status=ModelStatus.PRODUCTION)
        assert len(prods) == 1
        assert prods[0].version == "v1.0"


class TestRegistryPersistence:
    def test_saves_to_json(self, tmp_path: Path) -> None:
        path = tmp_path / "registry.json"
        reg = ModelRegistry(path)
        reg.register(_make_version())
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data["versions"]) == 1

    def test_loads_from_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "registry.json"
        # Create first instance
        reg1 = ModelRegistry(path)
        reg1.register(_make_version("v1.0"))
        # Load from same file
        reg2 = ModelRegistry(path)
        assert reg2.get("v1.0") is not None

    def test_empty_file_starts_fresh(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        assert len(reg.list_versions()) == 0


class TestRegistryPromotion:
    def test_promote_to_evaluating(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(_make_version("v1.0"))
        reg.promote_to_evaluating("v1.0")
        assert reg.get("v1.0").status == ModelStatus.EVALUATING

    def test_promote_to_canary(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(_make_version("v1.0", status=ModelStatus.EVALUATING))
        reg.promote_to_canary("v1.0")
        assert reg.get("v1.0").status == ModelStatus.CANARY

    def test_canary_replaces_old_canary(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(_make_version("v1.0", status=ModelStatus.CANARY))
        reg.register(_make_version("v2.0", status=ModelStatus.EVALUATING))
        reg.promote_to_canary("v2.0")
        assert reg.get("v1.0").status == ModelStatus.DEPRECATED
        assert reg.get("v2.0").status == ModelStatus.CANARY

    def test_promote_to_production(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(_make_version("v1.0", status=ModelStatus.CANARY))
        reg.promote_to_production("v1.0")
        assert reg.get("v1.0").status == ModelStatus.PRODUCTION

    def test_production_deprecates_old(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(_make_version("v1.0", status=ModelStatus.PRODUCTION))
        reg.register(_make_version("v2.0", status=ModelStatus.CANARY))
        reg.promote_to_production("v2.0")
        assert reg.get("v1.0").status == ModelStatus.DEPRECATED
        assert reg.get("v2.0").status == ModelStatus.PRODUCTION

    def test_get_production(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(_make_version("v1.0", status=ModelStatus.PRODUCTION))
        prod = reg.get_production()
        assert prod is not None
        assert prod.version == "v1.0"

    def test_get_production_none(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        assert reg.get_production() is None

    def test_mark_failed(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(_make_version("v1.0"))
        reg.mark_failed("v1.0")
        assert reg.get("v1.0").status == ModelStatus.FAILED

    def test_promote_unknown_raises(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        with pytest.raises(KeyError, match="not found"):
            reg.promote_to_production("nonexistent")


class TestRegistryRollback:
    def test_rollback_restores_deprecated(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(
            _make_version(
                "v1.0",
                status=ModelStatus.DEPRECATED,
                created_at="2026-02-20T00:00:00+00:00",
            )
        )
        reg.register(_make_version("v2.0", status=ModelStatus.PRODUCTION))
        restored = reg.rollback()
        assert restored is not None
        assert restored.version == "v1.0"
        assert reg.get("v1.0").status == ModelStatus.PRODUCTION
        assert reg.get("v2.0").status == ModelStatus.DEPRECATED

    def test_rollback_nothing_to_restore(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(_make_version("v1.0", status=ModelStatus.PRODUCTION))
        assert reg.rollback() is None

    def test_rollback_picks_most_recent(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(
            _make_version(
                "v1.0",
                status=ModelStatus.DEPRECATED,
                created_at="2026-02-18T00:00:00+00:00",
            )
        )
        reg.register(
            _make_version(
                "v1.5",
                status=ModelStatus.DEPRECATED,
                created_at="2026-02-22T00:00:00+00:00",
            )
        )
        reg.register(_make_version("v2.0", status=ModelStatus.PRODUCTION))
        restored = reg.rollback()
        assert restored.version == "v1.5"
