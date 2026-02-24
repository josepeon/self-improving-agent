"""Base provider abstraction — implement this for any model or API."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: list[str] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ProviderResponse:
    """Standardized response from any provider."""

    text: str
    provider: str
    model: str
    latency_ms: float
    tokens_used: int
    is_fallback: bool = False
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ProviderResponse(provider={self.provider!r}, "
            f"model={self.model!r}, "
            f"latency_ms={self.latency_ms:.1f}, "
            f"tokens={self.tokens_used}, "
            f"fallback={self.is_fallback})"
        )


class BaseModelProvider(ABC):
    """Abstract base class for all model providers.

    Implement this interface to add support for any LLM backend.
    Current implementations: GroqProvider, MLXProvider.
    Future: OpenAIProvider, AnthropicProvider, OllamaProvider, vLLMProvider.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'groq', 'mlx')."""
        ...

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> ProviderResponse:
        """Generate a complete response.

        Args:
            messages: Chat messages in OpenAI format
                [{"role": "system"|"user"|"assistant", "content": "..."}]
            config: Generation parameters (uses defaults if None).

        Returns:
            ProviderResponse with text and metadata.
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens.

        Args:
            messages: Chat messages in OpenAI format.
            config: Generation parameters.

        Yields:
            Individual tokens/chunks as strings.
        """
        ...
        # This yield is needed for the type checker to recognize this as
        # an async generator, even though subclasses provide the real impl.
        yield ""  # pragma: no cover

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if this provider is operational.

        Returns:
            True if the provider can serve requests.
        """
        ...
