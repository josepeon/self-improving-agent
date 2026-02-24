"""Tests for base provider abstraction."""

import pytest

from self_improving_agent.providers.base import (
    BaseModelProvider,
    GenerationConfig,
    ProviderResponse,
)


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_defaults(self):
        config = GenerationConfig()
        assert config.max_tokens == 150
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.stop_sequences is None

    def test_custom_values(self):
        config = GenerationConfig(
            max_tokens=500,
            temperature=0.3,
            top_p=0.95,
            stop_sequences=["<end>"],
        )
        assert config.max_tokens == 500
        assert config.temperature == 0.3
        assert config.stop_sequences == ["<end>"]

    def test_to_dict_excludes_none(self):
        config = GenerationConfig()
        d = config.to_dict()
        assert "stop_sequences" not in d
        assert "max_tokens" in d
        assert "temperature" in d

    def test_to_dict_includes_stop_sequences(self):
        config = GenerationConfig(stop_sequences=["<end>"])
        d = config.to_dict()
        assert d["stop_sequences"] == ["<end>"]


class TestProviderResponse:
    """Tests for ProviderResponse dataclass."""

    def test_creation(self):
        resp = ProviderResponse(
            text="Hello!",
            provider="test",
            model="test-model",
            latency_ms=42.5,
            tokens_used=10,
        )
        assert resp.text == "Hello!"
        assert resp.provider == "test"
        assert resp.model == "test-model"
        assert resp.latency_ms == 42.5
        assert resp.tokens_used == 10
        assert resp.is_fallback is False
        assert resp.metadata == {}

    def test_fallback_flag(self):
        resp = ProviderResponse(
            text="Hi",
            provider="groq",
            model="llama-70b",
            latency_ms=100.0,
            tokens_used=5,
            is_fallback=True,
        )
        assert resp.is_fallback is True

    def test_repr(self):
        resp = ProviderResponse(
            text="Hi",
            provider="mlx",
            model="llama-8b",
            latency_ms=50.0,
            tokens_used=3,
        )
        r = repr(resp)
        assert "mlx" in r
        assert "llama-8b" in r
        assert "50.0" in r


class TestBaseModelProviderInterface:
    """Tests that BaseModelProvider cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseModelProvider()

    def test_subclass_must_implement_all(self):
        """Subclass missing methods should fail to instantiate."""

        class IncompleteProvider(BaseModelProvider):
            @property
            def name(self):
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteProvider()
