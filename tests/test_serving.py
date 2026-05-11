"""T1.1 + T2.1: FastAPI inference server with SSE streaming.

We don't load MLX in CI — those tests are gated behind the 'slow' marker.
What this suite covers without MLX:
  - schema validation (OpenAI-compatible request/response)
  - / healthz and /v1/health endpoints
  - /v1/models listing from the registry
  - 503 when no production model is registered
  - SSE wire format (mocked provider)
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from self_improving_agent.providers.base import ProviderResponse
from self_improving_agent.serving import server as server_module
from self_improving_agent.serving.models import ChatCompletionRequest
from self_improving_agent.training.registry import (
    ModelRegistry,
    ModelStatus,
    ModelVersion,
)


# ---------------------------------------------------------------------
# Test-side fakes
# ---------------------------------------------------------------------


class FakeProvider:
    """Stand-in for MLXProvider — doesn't load weights."""

    def __init__(self, text: str = "hello world"):
        self.text = text
        self.calls: list[dict] = []

    async def generate(self, messages, config):
        self.calls.append({"messages": messages, "config": config})
        return ProviderResponse(
            text=self.text,
            provider="fake",
            model="test-model",
            latency_ms=1.0,
            tokens_used=3,
            metadata={"prompt_tokens": 5, "completion_tokens": 3},
        )

    async def generate_stream(self, messages, config):
        for token in self.text.split():
            yield token + " "


@pytest.fixture
def registry_with_prod(tmp_path):
    """Registry pre-populated with a production version."""
    registry_path = tmp_path / "registry.json"
    registry = ModelRegistry(registry_path)
    registry.register(
        ModelVersion(
            version="v1.0-test",
            base_model="test-base",
            adapter_path=None,
            status=ModelStatus.PRODUCTION,
        )
    )
    return registry_path


@pytest.fixture
def app_with_fake_provider(registry_with_prod, monkeypatch):
    """A test app that uses FakeProvider instead of MLX."""
    fake = FakeProvider()
    monkeypatch.setattr(server_module, "_build_provider", lambda v: fake)
    monkeypatch.setenv("SIA_REGISTRY_PATH", str(registry_with_prod))

    # Build a fresh app so the lifespan re-runs with the env we just set
    app = server_module._make_app()
    return app, fake


@pytest.fixture
def empty_app(tmp_path, monkeypatch):
    """An app whose registry has no production model."""
    monkeypatch.setenv("SIA_REGISTRY_PATH", str(tmp_path / "registry.json"))
    return server_module._make_app()


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------


class TestHealth:
    def test_liveness_ok(self, empty_app):
        with TestClient(empty_app) as client:
            response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_v1_health_loading_when_no_model(self, empty_app):
        with TestClient(empty_app) as client:
            response = client.get("/v1/health")
        assert response.status_code == 200
        body = response.json()
        # No model loaded yet — should report loading
        assert body["status"] in ("loading", "healthy")
        # version field always present
        assert "version" in body


# ---------------------------------------------------------------------
# Models endpoint
# ---------------------------------------------------------------------


class TestModelsEndpoint:
    def test_lists_versions_from_registry(self, app_with_fake_provider):
        app, _ = app_with_fake_provider
        with TestClient(app) as client:
            response = client.get("/v1/models")
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "list"
        ids = [m["id"] for m in body["data"]]
        assert "v1.0-test" in ids

    def test_empty_registry_returns_empty_list(self, empty_app):
        with TestClient(empty_app) as client:
            response = client.get("/v1/models")
        assert response.status_code == 200
        assert response.json()["data"] == []


# ---------------------------------------------------------------------
# Chat completions — non-streaming
# ---------------------------------------------------------------------


class TestChatCompletions:
    def test_503_when_no_production_model(self, empty_app):
        with TestClient(empty_app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "sia-production",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert response.status_code == 503

    def test_complete_response_shape(self, app_with_fake_provider):
        app, fake = app_with_fake_provider
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "sia-production",
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Say hello"},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 32,
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"] == "hello world"
        assert body["choices"][0]["finish_reason"] == "stop"
        # SIA extension
        assert "sia" in body
        assert body["sia"]["served_version"] == "v1.0-test"

        # Provider received the right messages + config
        assert fake.calls[0]["messages"][0]["role"] == "system"
        assert fake.calls[0]["config"].temperature == 0.3
        assert fake.calls[0]["config"].max_tokens == 32

    def test_invalid_request_400(self, app_with_fake_provider):
        app, _ = app_with_fake_provider
        with TestClient(app) as client:
            # Missing 'messages'
            response = client.post(
                "/v1/chat/completions",
                json={"model": "sia-production"},
            )
        assert response.status_code == 422  # fastapi default for validation


# ---------------------------------------------------------------------
# Chat completions — streaming (T2.1)
# ---------------------------------------------------------------------


class TestStreamingCompletions:
    def test_sse_wire_format(self, app_with_fake_provider):
        app, _ = app_with_fake_provider
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "sia-production",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = response.text

        # OpenAI-compatible: every frame starts with `data: `
        frames = [
            line[len("data: "):]
            for line in body.split("\n\n")
            if line.startswith("data: ")
        ]
        assert frames, "no SSE frames produced"

        # Last frame must be [DONE]
        assert frames[-1].strip() == "[DONE]"

        # Each non-[DONE] frame is valid JSON with object=chat.completion.chunk
        for frame in frames[:-1]:
            parsed = json.loads(frame)
            assert parsed["object"] == "chat.completion.chunk"
            assert "choices" in parsed

    def test_stream_includes_role_and_finish_reason(self, app_with_fake_provider):
        app, _ = app_with_fake_provider
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "sia-production",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        frames = [
            json.loads(line[len("data: "):])
            for line in response.text.split("\n\n")
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]
        # First frame should have role=assistant
        assert frames[0]["choices"][0]["delta"].get("role") == "assistant"
        # Final non-DONE frame should set finish_reason
        assert frames[-1]["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------
# Schema models
# ---------------------------------------------------------------------


class TestSchemas:
    def test_request_normalises_stop(self):
        # Just validates the pydantic schema accepts both shapes
        req1 = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "hi"}], stop="STOP",
        )
        req2 = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "hi"}],
            stop=["A", "B"],
        )
        assert req1.stop == "STOP"
        assert req2.stop == ["A", "B"]
