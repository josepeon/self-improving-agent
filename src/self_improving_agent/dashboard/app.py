"""Streamlit eval dashboard.

Reads the model registry and every evaluation report and renders:

  - registry overview table (versions, statuses, dataset sizes)
  - pass-rate-over-time chart per version
  - category-score heatmap for the latest report per version
  - compare A vs B view: pass-rate delta + per-category breakdown

Designed to be run locally during development. Auto-refreshes every 30s.

Run with:
    streamlit run src/self_improving_agent/dashboard/app.py

Or via the future ``sia dashboard`` CLI command.
"""

from __future__ import annotations

import os

import streamlit as st

from self_improving_agent.dashboard.data import (
    DashboardState,
    diff_versions,
    load_dashboard_state,
)


def main() -> None:
    st.set_page_config(
        page_title="SIA — Eval Dashboard",
        page_icon="📈",
        layout="wide",
    )

    registry_path = os.environ.get("SIA_REGISTRY_PATH", "models/registry.json")
    reports_dir = os.environ.get("SIA_REPORTS_DIR", "evaluation/reports")

    state = load_dashboard_state(registry_path, reports_dir)

    st.title("Self-Improving Agent — Eval Dashboard")
    st.caption(
        f"Registry: `{registry_path}`  •  Reports: `{reports_dir}`  •  "
        f"{len(state.versions)} version(s), {len(state.reports)} report(s)"
    )

    if not state.versions:
        st.warning(
            "No registry yet. Train a version with `sia train` or run one of "
            "the scripts/train_oh_model_*.py scripts to populate it."
        )
        return

    _render_overview(state)
    st.divider()
    _render_pass_rate_over_time(state)
    st.divider()
    _render_category_heatmap(state)
    st.divider()
    _render_compare_view(state)


# ---------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------


def _render_overview(state: DashboardState) -> None:
    st.subheader("Registry overview")
    prod = state.production_version()
    if prod:
        st.success(f"Production: **{prod.version}** (base: {prod.base_model})")
    else:
        st.info("No production model is currently registered.")

    rows = [
        {
            "Version": v.version,
            "Status": v.status,
            "Base model": v.base_model,
            "Dataset size": v.training_dataset_size,
            "Latest pass rate": (
                f"{v.latest_pass_rate:.0%}" if v.latest_pass_rate is not None else "—"
            ),
            "Created": (v.created_at or "")[:10],
        }
        for v in state.versions
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_pass_rate_over_time(state: DashboardState) -> None:
    st.subheader("Pass rate over time")
    if not state.reports:
        st.caption(
            "No reports yet. Run `sia evaluate --remote` or evaluate a "
            "registered version to populate evaluation/reports/."
        )
        return

    chart_data = []
    for r in state.reports:
        if r.pass_rate is None:
            continue
        chart_data.append(
            {
                "timestamp": r.timestamp or "",
                "model": r.model_version,
                "pass_rate": r.pass_rate,
            }
        )

    if not chart_data:
        st.caption("Reports exist but none have a pass rate.")
        return

    # Streamlit handles long-form well; group by model
    st.line_chart(
        chart_data,
        x="timestamp",
        y="pass_rate",
        color="model",
        height=320,
    )


def _render_category_heatmap(state: DashboardState) -> None:
    st.subheader("Latest category scores by version")
    rows: list[dict] = []
    for v in state.versions:
        if not v.reports:
            continue
        latest = v.reports[-1]
        for cat, score in latest.category_scores.items():
            rows.append(
                {
                    "version": v.version,
                    "category": cat,
                    "score": score,
                }
            )

    if not rows:
        st.caption(
            "No category scores yet — these populate once "
            "evaluation reports contain a category_scores field."
        )
        return

    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_compare_view(state: DashboardState) -> None:
    st.subheader("Compare two versions")
    names = [v.version for v in state.versions]
    if len(names) < 2:
        st.caption("Need at least two registered versions for compare view.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        version_a = st.selectbox("Version A", names, index=0)
    with col_b:
        version_b = st.selectbox("Version B", names, index=min(1, len(names) - 1))

    if version_a == version_b:
        st.info("Pick two different versions to compare.")
        return

    diff = diff_versions(state, version_a, version_b)
    delta = diff.get("pass_rate_delta")
    pass_a = diff.get("pass_rate_a")
    pass_b = diff.get("pass_rate_b")

    cols = st.columns(3)
    with cols[0]:
        st.metric(
            f"{version_a} pass rate", f"{pass_a:.0%}" if pass_a is not None else "—"
        )
    with cols[1]:
        st.metric(
            f"{version_b} pass rate", f"{pass_b:.0%}" if pass_b is not None else "—"
        )
    with cols[2]:
        if delta is not None:
            st.metric("Δ pass rate", f"{delta:+.0%}")
        else:
            st.metric("Δ pass rate", "—")

    if diff.get("categories"):
        st.markdown("**Category breakdown**")
        rows = []
        for cat, scores in diff["categories"].items():
            rows.append(
                {
                    "category": cat,
                    f"{version_a}": scores["a"],
                    f"{version_b}": scores["b"],
                    "delta": scores["delta"],
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
