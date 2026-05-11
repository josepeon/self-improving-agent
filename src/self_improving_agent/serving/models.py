"""OpenAI-compatible request/response schemas.

We mirror the subset of OpenAI's chat completions API that SIA actually
supports. Strictly compatible at the wire level for: messages format,
model, stream, temperature, max_tokens, top_p, stop, plus the SSE
``data: {...}\\n\\n`` frame shape with ``[DONE]`` terminator.

Things deliberately not implemented: tools/function calling, logit_bias,
n>1 completions, system_fingerprint. Add if a real client needs them.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(
        ...,
        description=(
            "Model identifier. Special value 'sia-production' resolves to "
            "the currently-promoted version. Specific version tags from the "
            "registry are also accepted."
        ),
    )
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    stream: bool = False
    stop: list[str] | str | None = None


# ---------------------------------------------------------------------
# Responses (non-streaming)
# ---------------------------------------------------------------------


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class CompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage = Field(default_factory=CompletionUsage)
    # SIA-specific extension — clients can ignore.
    sia: dict | None = None


# ---------------------------------------------------------------------
# Responses (streaming, OpenAI SSE shape)
# ---------------------------------------------------------------------


class ChatCompletionDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatCompletionStreamChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionDelta
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamChoice]


# ---------------------------------------------------------------------
# Health / metadata
# ---------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["healthy", "loading", "degraded"] = "healthy"
    model: str | None = None
    version: str = "0.1.0"
    routing: dict | None = None
