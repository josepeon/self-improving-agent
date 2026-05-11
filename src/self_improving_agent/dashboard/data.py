"""Pure data layer for the eval dashboard.

Separated from app.py so it's importable + testable without Streamlit.
``load_dashboard_state()`` reads the registry, every report in the
reports directory, and produces a normalised in-memory view.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReportSummary:
    """One evaluation report — flattened for display."""

    file: str
    model_version: str
    timestamp: str | None = None
    pass_rate: float | None = None
    passed_tests: int = 0
    total_tests: int = 0
    avg_latency_ms: float | None = None
    category_scores: dict[str, float] = field(default_factory=dict)
    mtime: float = 0.0


@dataclass
class VersionSummary:
    """One registry version + its derived eval state."""

    version: str
    status: str
    base_model: str
    adapter_path: str | None
    created_at: str | None
    training_dataset_size: int
    reports: list[ReportSummary] = field(default_factory=list)

    @property
    def latest_pass_rate(self) -> float | None:
        if not self.reports:
            return None
        return self.reports[-1].pass_rate


@dataclass
class DashboardState:
    versions: list[VersionSummary]
    reports: list[ReportSummary]

    def production_version(self) -> VersionSummary | None:
        for v in self.versions:
            if v.status == "production":
                return v
        return None


def load_dashboard_state(
    registry_path: str | Path = "models/registry.json",
    reports_dir: str | Path = "evaluation/reports",
) -> DashboardState:
    """Read everything off disk and assemble a dashboard view."""
    versions = _load_versions(Path(registry_path))
    reports = _load_reports(Path(reports_dir))

    # Wire each report into its matching version
    by_version: dict[str, list[ReportSummary]] = {}
    for r in reports:
        by_version.setdefault(r.model_version, []).append(r)
    for v in versions:
        v.reports = sorted(
            by_version.get(v.version, []), key=lambda r: r.mtime
        )

    return DashboardState(versions=versions, reports=reports)


def _load_versions(registry_path: Path) -> list[VersionSummary]:
    if not registry_path.exists():
        return []
    try:
        with open(registry_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    versions: list[VersionSummary] = []
    for entry in data.get("versions", []):
        versions.append(VersionSummary(
            version=entry.get("version", "<unknown>"),
            status=entry.get("status", "unknown"),
            base_model=entry.get("base_model", ""),
            adapter_path=entry.get("adapter_path"),
            created_at=entry.get("created_at"),
            training_dataset_size=int(entry.get("training_dataset_size", 0)),
        ))
    return versions


def _load_reports(reports_dir: Path) -> list[ReportSummary]:
    if not reports_dir.exists():
        return []
    summaries: list[ReportSummary] = []
    for path in sorted(reports_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        summaries.append(ReportSummary(
            file=str(path),
            model_version=data.get("model_version", "<unknown>"),
            timestamp=data.get("timestamp"),
            pass_rate=data.get("pass_rate"),
            passed_tests=int(data.get("passed_tests", 0)),
            total_tests=int(data.get("total_tests", 0)),
            avg_latency_ms=data.get("avg_latency_ms"),
            category_scores=data.get("category_scores", {}),
            mtime=path.stat().st_mtime,
        ))
    return summaries


def diff_versions(
    state: DashboardState, version_a: str, version_b: str
) -> dict:
    """Compare two versions' latest reports — used by the "compare A vs B" view."""
    def _by_name(name: str) -> VersionSummary | None:
        return next((v for v in state.versions if v.version == name), None)

    a = _by_name(version_a)
    b = _by_name(version_b)
    if not a or not b:
        return {"error": "version not found"}

    a_latest = a.reports[-1] if a.reports else None
    b_latest = b.reports[-1] if b.reports else None

    return {
        "version_a": version_a,
        "version_b": version_b,
        "pass_rate_a": a_latest.pass_rate if a_latest else None,
        "pass_rate_b": b_latest.pass_rate if b_latest else None,
        "pass_rate_delta": (
            (b_latest.pass_rate - a_latest.pass_rate)
            if a_latest and b_latest
            and a_latest.pass_rate is not None
            and b_latest.pass_rate is not None
            else None
        ),
        "categories": _category_diff(a_latest, b_latest),
    }


def _category_diff(
    a: ReportSummary | None, b: ReportSummary | None
) -> dict[str, dict]:
    if not a or not b:
        return {}
    out: dict[str, dict] = {}
    all_cats = set(a.category_scores) | set(b.category_scores)
    for cat in sorted(all_cats):
        a_score = a.category_scores.get(cat)
        b_score = b.category_scores.get(cat)
        out[cat] = {
            "a": a_score,
            "b": b_score,
            "delta": (b_score - a_score) if a_score is not None and b_score is not None else None,
        }
    return out
