"""Model providers — unified interface for any LLM backend."""

from self_improving_agent.providers.base import (
    BaseModelProvider,
    GenerationConfig,
    ProviderResponse,
)
from self_improving_agent.providers.groq_provider import GroqProvider
from self_improving_agent.providers.mlx_provider import MLXProvider
from self_improving_agent.providers.router import ModelRouter

__all__ = [
    "BaseModelProvider",
    "GenerationConfig",
    "GroqProvider",
    "MLXProvider",
    "ModelRouter",
    "ProviderResponse",
]
