"""
Prometheus-compatible metrics

Hand-rolled metric formatting — we don't pull in `prometheus_client` as
a base dependency because the surface we need is small and an extra
dep on the serving path isn't worth it.

Exposes:
  - Counter: monotonic count, labelled
  - Histogram: bucketed latency histograms (Prometheus expected schema)
  - Gauge: instantaneous value
  - render(metrics): Prometheus text-format string

Used by the inference server (T1.1)'s /metrics endpoint, and by the
training pipeline for one-shot summaries.

Format reference:
    https://prometheus.io/docs/concepts/data_model/
    https://prometheus.io/docs/instrumenting/exposition_formats/

Naming convention follows Prometheus suggestions:
  sia_requests_total{provider,model}
  sia_request_duration_seconds{provider,model}
  sia_fallbacks_total{from,to}
  sia_version_info{version,status}
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable


# Histogram buckets: 50ms..30s, geometric ratio. Tuned for LLM inference
# where p50 is typically 200-800ms and p99 can stretch to ~10s.
DEFAULT_BUCKETS = (
    0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0,
)


def _format_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    items = ",".join(
        f'{k}="{_escape_value(str(v))}"'
        for k, v in sorted(labels.items())
    )
    return "{" + items + "}"


def _escape_value(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# ---------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------


class Counter:
    """Monotonic counter labelled by zero or more dimensions."""

    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help_text = help_text
        self._lock = threading.Lock()
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)

    def inc(self, value: float = 1.0, **labels: str) -> None:
        if value < 0:
            raise ValueError("Counter cannot be decremented")
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] += value

    def render(self) -> str:
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            for key, value in self._values.items():
                labels = dict(key)
                lines.append(f"{self.name}{_format_labels(labels)} {value}")
        return "\n".join(lines)


# ---------------------------------------------------------------------
# Gauge
# ---------------------------------------------------------------------


class Gauge:
    """Instantaneous value, labelled."""

    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help_text = help_text
        self._lock = threading.Lock()
        self._values: dict[tuple[tuple[str, str], ...], float] = {}

    def set(self, value: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = value

    def render(self) -> str:
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} gauge",
        ]
        with self._lock:
            for key, value in self._values.items():
                labels = dict(key)
                lines.append(f"{self.name}{_format_labels(labels)} {value}")
        return "\n".join(lines)


# ---------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------


@dataclass
class _HistogramSample:
    sum: float = 0.0
    count: int = 0
    bucket_counts: dict[float, int] = field(default_factory=dict)

    def observe(self, value: float, buckets: Iterable[float]) -> None:
        self.sum += value
        self.count += 1
        for bucket in buckets:
            if value <= bucket:
                self.bucket_counts[bucket] = self.bucket_counts.get(bucket, 0) + 1


class Histogram:
    """Bucketed histogram. Produces Prometheus-style _bucket, _sum, _count series."""

    def __init__(
        self,
        name: str,
        help_text: str,
        buckets: Iterable[float] = DEFAULT_BUCKETS,
    ):
        self.name = name
        self.help_text = help_text
        self.buckets = tuple(sorted(buckets))
        self._lock = threading.Lock()
        self._samples: dict[tuple[tuple[str, str], ...], _HistogramSample] = (
            defaultdict(_HistogramSample)
        )

    def observe(self, value: float, **labels: str) -> None:
        if math.isnan(value) or math.isinf(value):
            return
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._samples[key].observe(value, self.buckets)

    def render(self) -> str:
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            for key, sample in self._samples.items():
                labels = dict(key)
                # Buckets — cumulative
                cumulative = 0
                for bucket in self.buckets:
                    cumulative = sample.bucket_counts.get(bucket, 0)
                    bucket_labels = {**labels, "le": _fmt_le(bucket)}
                    lines.append(
                        f"{self.name}_bucket{_format_labels(bucket_labels)} {cumulative}"
                    )
                # +Inf bucket = total count
                lines.append(
                    f"{self.name}_bucket"
                    f"{_format_labels({**labels, 'le': '+Inf'})} {sample.count}"
                )
                lines.append(f"{self.name}_sum{_format_labels(labels)} {sample.sum}")
                lines.append(
                    f"{self.name}_count{_format_labels(labels)} {sample.count}"
                )
        return "\n".join(lines)


def _fmt_le(bucket: float) -> str:
    """Format a bucket boundary for Prometheus."""
    if bucket == int(bucket):
        return str(int(bucket))
    return str(bucket)


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------


class MetricRegistry:
    """Collection of metrics. Renders them all in Prometheus text format."""

    def __init__(self) -> None:
        self._metrics: list[Counter | Gauge | Histogram] = []

    def register(self, metric: Counter | Gauge | Histogram) -> None:
        self._metrics.append(metric)

    def render(self) -> str:
        return "\n\n".join(m.render() for m in self._metrics) + "\n"


# ---------------------------------------------------------------------
# SIA-specific metrics
# ---------------------------------------------------------------------


def make_sia_registry() -> tuple[MetricRegistry, dict]:
    """Construct the default SIA metric set + return a dict of named handles.

    Returned handles:
        requests_total      Counter[provider, model, status]
        request_duration    Histogram[provider, model]
        fallbacks_total     Counter[from, to]
        tokens_total        Counter[provider, model, kind]
        version_info        Gauge[version, status]
    """
    registry = MetricRegistry()

    requests_total = Counter(
        "sia_requests_total",
        "Total number of inference requests handled, by provider/model/status.",
    )
    request_duration = Histogram(
        "sia_request_duration_seconds",
        "End-to-end request duration in seconds, by provider/model.",
    )
    fallbacks_total = Counter(
        "sia_fallbacks_total",
        "Times the router fell back from one provider to another.",
    )
    tokens_total = Counter(
        "sia_tokens_total",
        "Total tokens processed, by provider/model/kind (prompt|completion).",
    )
    version_info = Gauge(
        "sia_version_info",
        "Loaded model version (value=1 indicates currently loaded).",
    )

    for m in (requests_total, request_duration, fallbacks_total, tokens_total, version_info):
        registry.register(m)

    return registry, {
        "requests_total": requests_total,
        "request_duration": request_duration,
        "fallbacks_total": fallbacks_total,
        "tokens_total": tokens_total,
        "version_info": version_info,
    }
