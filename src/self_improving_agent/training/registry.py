"""Model version registry - track, promote, and rollback model versions.

Persists a lightweight JSON manifest alongside trained adapters so the
pipeline knows which version is ``production``, ``canary``, etc.

Promotion flow::

    training -> evaluating -> canary (5% traffic) -> production
                                                        |
                                                   deprecated
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class ModelStatus(StrEnum):
    """Lifecycle stages for a model version."""

    TRAINING = "training"
    EVALUATING = "evaluating"
    CANARY = "canary"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    FAILED = "failed"


@dataclass
class ModelVersion:
    """Metadata for a single model version."""

    version: str  # e.g. "v1.0", "v1.1"
    base_model: str
    adapter_path: str | None = None  # None for the untuned base
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    training_dataset_size: int = 0
    evaluation_report_path: str | None = None
    evaluation_pass_rate: float | None = None
    status: str = ModelStatus.TRAINING
    # Lineage tracking: what data sources contributed to this version
    data_sources: dict[str, Any] = field(default_factory=dict)
    parent_version: str | None = None

    # ── Serialisation helpers ─────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelVersion:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ModelRegistry:
    """Track model versions and manage promotion.

    State is persisted to a single JSON file (``registry.json``)
    containing a list of :class:`ModelVersion` entries.
    """

    def __init__(self, registry_path: str | Path = "models/registry.json") -> None:
        self.registry_path = Path(registry_path)
        self.versions: dict[str, ModelVersion] = self._load()

    # ── CRUD ──────────────────────────────────────────────────────────

    def register(self, version: ModelVersion) -> None:
        """Register a new model version (or overwrite existing).

        Registering a version directly as ``production`` or ``canary``
        demotes any other holder of that status — the registry keeps at
        most one of each.
        """
        if version.status in (ModelStatus.PRODUCTION, ModelStatus.CANARY):
            self._demote_others(version.version, version.status)
        self.versions[version.version] = version
        self._save()

    def get(self, version: str) -> ModelVersion | None:
        """Retrieve a specific version by tag."""
        return self.versions.get(version)

    def list_versions(self, *, status: str | None = None) -> list[ModelVersion]:
        """List all registered versions, optionally filtered by status."""
        if status is None:
            return list(self.versions.values())
        return [v for v in self.versions.values() if v.status == status]

    # ── Promotion ─────────────────────────────────────────────────────

    def get_production(self) -> ModelVersion | None:
        """Return the current production model, if any.

        If legacy data contains several production entries, the newest
        by ``created_at`` wins (deterministically).
        """
        prods = self.list_versions(status=ModelStatus.PRODUCTION)
        return max(prods, key=lambda v: v.created_at) if prods else None

    def get_canary(self) -> ModelVersion | None:
        """Return the current canary model, if any."""
        canaries = self.list_versions(status=ModelStatus.CANARY)
        return max(canaries, key=lambda v: v.created_at) if canaries else None

    def promote_to_evaluating(self, version: str) -> None:
        """Move a version from training to evaluating."""
        self._transition(version, ModelStatus.EVALUATING)

    def promote_to_canary(self, version: str) -> None:
        """Move a version to canary deployment.

        If another canary already exists it is demoted to deprecated.
        """
        self._demote_others(version, ModelStatus.CANARY)
        self._transition(version, ModelStatus.CANARY)

    def promote_to_production(self, version: str) -> None:
        """Promote a (canary) version to production.

        The previously-production version is deprecated automatically.
        """
        self._demote_others(version, ModelStatus.PRODUCTION)
        self._transition(version, ModelStatus.PRODUCTION)

    def _demote_others(self, version: str, status: str) -> None:
        """Deprecate every OTHER version currently holding ``status``."""
        for other in self.list_versions(status=status):
            if other.version != version:
                other.status = ModelStatus.DEPRECATED

    def mark_failed(self, version: str) -> None:
        """Mark a version that did not pass evaluation."""
        self._transition(version, ModelStatus.FAILED)

    def rollback(self) -> ModelVersion | None:
        """Rollback: deprecate current production and restore the most
        recently-deprecated version.

        Returns the newly-promoted version or ``None`` if there is
        nothing to roll back to.
        """
        current = self.get_production()
        # Find the most recent deprecated version
        deprecated = sorted(
            self.list_versions(status=ModelStatus.DEPRECATED),
            key=lambda v: v.created_at,
            reverse=True,
        )
        if not deprecated:
            return None

        restored = deprecated[0]
        if current:
            current.status = ModelStatus.DEPRECATED
        restored.status = ModelStatus.PRODUCTION
        self._save()
        return restored

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> dict[str, ModelVersion]:
        if not self.registry_path.exists():
            return {}
        with open(self.registry_path) as f:
            data = json.load(f)
        return {
            v["version"]: ModelVersion.from_dict(v) for v in data.get("versions", [])
        }

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now(UTC).isoformat(),
            "versions": [v.to_dict() for v in self.versions.values()],
        }
        # Atomic write: the serving layer polls this file for hot-reload,
        # so it must never observe a half-written JSON.
        tmp_path = self.registry_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, self.registry_path)

    def _transition(self, version: str, new_status: str) -> None:
        entry = self.versions.get(version)
        if entry is None:
            msg = f"Version '{version}' not found in registry"
            raise KeyError(msg)
        entry.status = new_status
        self._save()
