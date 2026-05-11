"""FastAPI inference server.

Endpoints:

  GET  /healthz                          — liveness; returns immediately
  GET  /v1/health                        — current model + routing stats
  GET  /v1/models                        — list registry-known versions
  POST /v1/chat/completions              — main inference endpoint
                                            (stream=true → SSE)

Model resolution:

  - ``model: "sia-production"``   → registry's current production version
  - ``model: "sia-canary"``       → registry's current canary version
  - any other id                   → looked up by version tag in the registry

A registry watch task reloads the production adapter without a restart
when ``models/registry.json`` changes on disk. Polling cadence: 30s.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from self_improving_agent.feedback.store import (
    Feedback,
    FeedbackRating,
    FeedbackStore,
)
from self_improving_agent.providers.base import GenerationConfig
from self_improving_agent.serving.models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatMessage,
    CompletionUsage,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
)
from self_improving_agent.training.registry import ModelRegistry, ModelVersion

logger = logging.getLogger(__name__)

REGISTRY_POLL_SECONDS = 30


# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------


class ServerState:
    """Owned process state — registry, current provider, last-mtime watcher."""

    def __init__(
        self,
        registry_path: str | Path,
        feedback_path: str | Path = "data/feedback/feedback.jsonl",
    ) -> None:
        self.registry_path = Path(registry_path)
        self.registry = ModelRegistry(self.registry_path)
        self.feedback = FeedbackStore(feedback_path)
        self._provider: Any | None = None
        self._loaded_version: str | None = None
        self._mtime: float = 0.0
        self._reload_lock = asyncio.Lock()

    @property
    def loaded_version(self) -> str | None:
        return self._loaded_version

    async def ensure_loaded(self) -> Any:
        """Ensure the current production provider is loaded; reload if changed."""
        async with self._reload_lock:
            target = self.registry.get_production()
            if target is None:
                raise HTTPException(
                    status_code=503,
                    detail="No production model registered.",
                )

            if (
                self._provider is not None
                and self._loaded_version == target.version
            ):
                return self._provider

            logger.info("Loading model %s (adapter=%s)",
                        target.version, target.adapter_path)
            self._provider = _build_provider(target)
            self._loaded_version = target.version
            return self._provider

    async def watch_registry(self) -> None:
        """Background task — reload when registry mtime changes."""
        while True:
            try:
                if self.registry_path.exists():
                    mtime = self.registry_path.stat().st_mtime
                    if mtime != self._mtime:
                        self._mtime = mtime
                        # Re-read; ensure_loaded() will swap if version changed
                        self.registry = ModelRegistry(self.registry_path)
                        prod = self.registry.get_production()
                        if prod and prod.version != self._loaded_version:
                            logger.info(
                                "Registry changed: hot-reloading %s -> %s",
                                self._loaded_version, prod.version,
                            )
                            self._provider = None
                            self._loaded_version = None
            except Exception as exc:  # noqa: BLE001
                logger.warning("Registry watch error: %s", exc)
            await asyncio.sleep(REGISTRY_POLL_SECONDS)


def _build_provider(version: ModelVersion) -> Any:
    """Instantiate the right provider for ``version``.

    Currently MLX-only since SIA's local target is an MLX model. Future:
    inspect ``version.metadata`` for ``provider: groq|openai|...`` and
    dispatch.
    """
    from self_improving_agent.providers.mlx_provider import MLXProvider

    return MLXProvider(
        adapter_path=version.adapter_path,
        model_path=version.base_model,
    )


# ---------------------------------------------------------------------
# App factory + lifespan
# ---------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    state = ServerState(
        registry_path=os.environ.get("SIA_REGISTRY_PATH", "models/registry.json"),
        feedback_path=os.environ.get(
            "SIA_FEEDBACK_PATH", "data/feedback/feedback.jsonl"
        ),
    )
    app.state.sia = state

    # Kick off the registry watcher — but don't block startup if no model is
    # registered yet.
    watch_task = asyncio.create_task(state.watch_registry())
    try:
        yield
    finally:
        watch_task.cancel()
        try:
            await watch_task
        except asyncio.CancelledError:
            pass


def _make_app() -> FastAPI:
    app = FastAPI(
        title="Self-Improving Agent",
        description="OpenAI-compatible inference API for SIA-trained models.",
        version="0.1.0",
        lifespan=_lifespan,
    )

    @app.get("/healthz")
    async def healthz() -> dict:
        # Liveness only — does not block on model load
        return {"status": "ok"}

    @app.get("/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        state: ServerState = app.state.sia
        return HealthResponse(
            status="healthy" if state.loaded_version else "loading",
            model=state.loaded_version,
            routing=None,  # router stats land here when T1.4 wires it up
        )

    @app.get("/v1/models")
    async def list_models() -> dict:
        state: ServerState = app.state.sia
        versions = state.registry.list_versions()
        return {
            "object": "list",
            "data": [
                {
                    "id": v.version,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "sia",
                    "status": v.status,
                }
                for v in versions
            ],
        }

    @app.post("/v1/feedback", response_model=FeedbackResponse)
    async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
        """Record operator/user feedback on a production response.

        The interaction_id should be the id of the original /v1/chat/completions
        response. Submitting the same (interaction_id, rating, correction)
        triple twice is a no-op.
        """
        state: ServerState = app.state.sia
        feedback = Feedback(
            interaction_id=request.interaction_id,
            rating=FeedbackRating(request.rating),
            user_message=request.user_message,
            original_response=request.original_response,
            correction=request.correction,
            model_version=request.model_version,
            metadata=request.metadata or {},
        )
        accepted = state.feedback.add(feedback)
        return FeedbackResponse(id=feedback.id, accepted=accepted)

    @app.get("/v1/feedback/stats")
    async def feedback_stats() -> dict:
        state: ServerState = app.state.sia
        return {
            "totals": state.feedback.counts_by_rating(),
            "by_version": state.feedback.counts_by_version(),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        state: ServerState = app.state.sia
        provider = await state.ensure_loaded()

        if request.stream:
            return StreamingResponse(
                _stream_chat(provider, request, state.loaded_version),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        return await _complete_chat(provider, request, state.loaded_version)

    return app


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------


async def _complete_chat(
    provider: Any,
    request: ChatCompletionRequest,
    served_version: str | None,
) -> ChatCompletionResponse:
    config = GenerationConfig(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop_sequences=_normalize_stop(request.stop),
    )
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    started = time.perf_counter()
    response = await provider.generate(messages, config)
    elapsed_ms = (time.perf_counter() - started) * 1000

    return ChatCompletionResponse(
        model=served_version or request.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=response.text),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=response.metadata.get("prompt_tokens", 0)
            if response.metadata else 0,
            completion_tokens=response.metadata.get("completion_tokens", 0)
            if response.metadata else 0,
            total_tokens=response.tokens_used,
        ),
        sia={
            "latency_ms": round(elapsed_ms, 2),
            "provider": response.provider,
            "served_version": served_version,
        },
    )


async def _stream_chat(
    provider: Any,
    request: ChatCompletionRequest,
    served_version: str | None,
):
    """SSE generator matching OpenAI's wire format."""
    import json
    import uuid

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    model_label = served_version or request.model

    # First chunk: role marker
    first = ChatCompletionChunk(
        id=completion_id,
        model=model_label,
        choices=[
            ChatCompletionStreamChoice(
                delta=ChatCompletionDelta(role="assistant", content=""),
            )
        ],
    )
    yield f"data: {first.model_dump_json()}\n\n"

    config = GenerationConfig(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop_sequences=_normalize_stop(request.stop),
    )
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        async for token in provider.generate_stream(messages, config):
            chunk = ChatCompletionChunk(
                id=completion_id,
                model=model_label,
                choices=[
                    ChatCompletionStreamChoice(
                        delta=ChatCompletionDelta(content=token),
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
    except Exception as exc:  # noqa: BLE001
        error_payload = {"error": {"message": str(exc), "type": type(exc).__name__}}
        yield f"data: {json.dumps(error_payload)}\n\n"

    # Final stop chunk
    done = ChatCompletionChunk(
        id=completion_id,
        model=model_label,
        choices=[
            ChatCompletionStreamChoice(
                delta=ChatCompletionDelta(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {done.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


def _normalize_stop(stop):
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return list(stop)


# Module-level app — uvicorn imports this directly.
app = _make_app()
