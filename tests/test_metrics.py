"""T3.5: Prometheus-format metrics."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from self_improving_agent.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricRegistry,
    make_sia_registry,
)
from self_improving_agent.providers.base import ProviderResponse
from self_improving_agent.serving import server as server_module
from self_improving_agent.training.registry import (
    ModelRegistry,
    ModelStatus,
    ModelVersion,
)

# ---------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------


class TestCounter:
    def test_basic_inc(self):
        c = Counter("x", "test")
        c.inc()
        c.inc(2.5)
        rendered = c.render()
        assert "x 3.5" in rendered
        assert "# TYPE x counter" in rendered

    def test_labels_are_sorted(self):
        c = Counter("x", "test")
        c.inc(provider="groq", model="llama")
        rendered = c.render()
        assert 'model="llama"' in rendered
        assert 'provider="groq"' in rendered

    def test_negative_inc_rejected(self):
        c = Counter("x", "test")
        with pytest.raises(ValueError, match="cannot be decremented"):
            c.inc(-1)

    def test_independent_label_series(self):
        c = Counter("x", "test")
        c.inc(provider="groq")
        c.inc(provider="mlx")
        rendered = c.render()
        assert rendered.count("x{provider=") == 2

    def test_label_escape(self):
        c = Counter("x", "test")
        c.inc(message='has "quotes" and\nnewlines')
        rendered = c.render()
        assert '\\"quotes\\"' in rendered
        assert "\\n" in rendered


# ---------------------------------------------------------------------
# Gauge
# ---------------------------------------------------------------------


class TestGauge:
    def test_set_and_render(self):
        g = Gauge("memory_bytes", "test")
        g.set(1024)
        assert "memory_bytes 1024" in g.render()

    def test_set_overwrites(self):
        g = Gauge("memory_bytes", "test")
        g.set(100)
        g.set(200)
        rendered = g.render()
        assert "memory_bytes 200" in rendered
        assert "100" not in rendered.split("\n")[-1]

    def test_labelled(self):
        g = Gauge("version_info", "test")
        g.set(1, version="v1.0", status="loaded")
        rendered = g.render()
        assert 'version="v1.0"' in rendered
        assert 'status="loaded"' in rendered


# ---------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------


class TestHistogram:
    def test_records_count_and_sum(self):
        h = Histogram("latency", "test")
        for v in [0.1, 0.5, 1.0, 2.0]:
            h.observe(v)
        rendered = h.render()
        assert "latency_count 4" in rendered
        assert "latency_sum 3.6" in rendered

    def test_buckets_are_cumulative(self):
        h = Histogram("latency", "test", buckets=[0.1, 0.5, 1.0])
        for v in [0.05, 0.3, 0.7, 1.5]:
            h.observe(v)
        rendered = h.render()
        # le=0.1: just the 0.05
        assert re.search(r'latency_bucket{le="0.1"} 1', rendered)
        # le=1.0: 0.05, 0.3, 0.7 = 3
        assert re.search(r'latency_bucket{le="1"} 3', rendered)
        # +Inf: total
        assert re.search(r'latency_bucket{le="\+Inf"} 4', rendered)

    def test_inf_and_nan_ignored(self):
        h = Histogram("latency", "test")
        h.observe(float("nan"))
        h.observe(float("inf"))
        h.observe(1.0)
        rendered = h.render()
        assert "latency_count 1" in rendered


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------


class TestRegistry:
    def test_renders_each_metric(self):
        registry = MetricRegistry()
        registry.register(Counter("a", "ha"))
        registry.register(Gauge("b", "hb"))
        out = registry.render()
        assert "# HELP a ha" in out
        assert "# HELP b hb" in out

    def test_sia_factory_returns_expected_handles(self):
        registry, handles = make_sia_registry()
        for key in (
            "requests_total",
            "request_duration",
            "fallbacks_total",
            "tokens_total",
            "version_info",
        ):
            assert key in handles
        # Initial render is well-formed even with no data
        rendered = registry.render()
        assert "sia_requests_total" in rendered


# ---------------------------------------------------------------------
# /metrics endpoint
# ---------------------------------------------------------------------


@pytest.fixture
def server_app(tmp_path, monkeypatch):
    """Server with FakeProvider and a populated registry."""

    class FakeProvider:
        name = "fake"

        async def generate(self, messages, config):
            return ProviderResponse(
                text="ok",
                provider="fake",
                model="m",
                latency_ms=0.0,
                tokens_used=1,
            )

        async def generate_stream(self, *_a, **_kw):  # pragma: no cover
            if False:
                yield ""

        async def health_check(self) -> bool:  # pragma: no cover
            return True

    registry_path = tmp_path / "registry.json"
    registry = ModelRegistry(registry_path)
    registry.register(
        ModelVersion(
            version="v1.0-test",
            base_model="b",
            status=ModelStatus.PRODUCTION,
        )
    )
    monkeypatch.setattr(server_module, "_build_provider", lambda v: FakeProvider())
    monkeypatch.setenv("SIA_REGISTRY_PATH", str(registry_path))
    monkeypatch.setenv("SIA_FEEDBACK_PATH", str(tmp_path / "feedback.jsonl"))
    return server_module._make_app()


class TestMetricsEndpoint:
    def test_serves_prometheus_text(self, server_app):
        with TestClient(server_app) as client:
            response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        body = response.text
        # Each base metric appears at least as a # HELP line
        for metric in (
            "sia_requests_total",
            "sia_request_duration_seconds",
            "sia_fallbacks_total",
            "sia_tokens_total",
            "sia_version_info",
        ):
            assert f"# HELP {metric}" in body

    def test_version_info_reflects_loaded_model_after_request(self, server_app):
        with TestClient(server_app) as client:
            # First trigger a chat completion so the model gets loaded
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "sia-production",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            body = client.get("/metrics").text
        assert 'sia_version_info{status="loaded",version="v1.0-test"} 1' in body
