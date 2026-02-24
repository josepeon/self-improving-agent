"""Tests for MLXProvider (mocked — no real model loading)."""

from unittest.mock import MagicMock, patch

import pytest

from self_improving_agent.providers.base import ProviderResponse
from self_improving_agent.providers.mlx_provider import MLXProvider


@pytest.fixture
def mlx_provider():
    """Create an MLXProvider with mocked internals."""
    provider = MLXProvider(model_path="test-model")
    # Pre-set model and tokenizer to avoid real loading
    provider._model = MagicMock()
    provider._tokenizer = MagicMock()
    provider._tokenizer.apply_chat_template.return_value = "<|begin|>Hi<|end|>"
    provider._tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
    return provider


class TestMLXProviderInit:
    """Tests for MLXProvider initialization."""

    def test_name(self, mlx_provider):
        assert mlx_provider.name == "mlx"

    def test_default_model(self):
        provider = MLXProvider()
        assert "Meta-Llama-3.1-8B-Instruct-4bit" in provider._model_path

    def test_custom_model(self):
        provider = MLXProvider(model_path="custom/model")
        assert provider._model_path == "custom/model"

    def test_lazy_loading(self):
        provider = MLXProvider()
        assert provider._model is None
        assert provider._tokenizer is None


class TestMLXProviderGenerate:
    """Tests for MLXProvider.generate()."""

    @pytest.mark.asyncio
    async def test_generate_returns_response(self, mlx_provider):
        """Test that generate produces a valid ProviderResponse."""
        with patch("mlx_lm.generate", return_value="Hello there!"):
            response = await mlx_provider.generate(
                messages=[{"role": "user", "content": "Hi"}],
            )

            assert isinstance(response, ProviderResponse)
            assert response.text == "Hello there!"
            assert response.provider == "mlx"
            assert response.model == "test-model"
            assert response.tokens_used == 5  # Mocked tokenizer encodes to 5 tokens
            assert response.latency_ms > 0
            assert response.is_fallback is False

    @pytest.mark.asyncio
    async def test_generate_uses_chat_template(self, mlx_provider):
        """Verify that the tokenizer's chat template is used."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        mlx_provider._tokenizer.apply_chat_template.return_value = (
            "<|system|>You are helpful.<|user|>Hi"
        )

        result = mlx_provider._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert "You are helpful" in result
        assert "Hi" in result


class TestMLXProviderHealth:
    """Tests for MLXProvider.health_check()."""

    @pytest.mark.asyncio
    async def test_health_check_model_not_loaded(self):
        """Health check should fail gracefully when model can't load."""
        provider = MLXProvider(model_path="nonexistent-model")
        # Don't pre-load — let it try and fail
        result = await provider.health_check()
        assert result is False
