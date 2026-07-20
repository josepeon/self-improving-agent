"""Tests for MLXProvider (mocked — no real model loading)."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from self_improving_agent.providers.base import ProviderResponse
from self_improving_agent.providers.mlx_provider import MLXProvider

# Ensure mlx_lm is importable even on Linux CI (where it doesn't exist).
# We inject a stub module so that patch("mlx_lm.X") works everywhere.
if "mlx_lm" not in sys.modules:
    _stub = ModuleType("mlx_lm")
    _stub.load = MagicMock()
    _stub.generate = MagicMock()
    sys.modules["mlx_lm"] = _stub


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
            # Tokenizer encodes prompt AND response (5 each in this mock)
            assert response.tokens_used == 10
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
        with patch("mlx_lm.load", side_effect=RuntimeError("model not found")):
            result = await provider.health_check()
        assert result is False


class TestMLXProviderEnsureLoaded:
    """Tests for lazy model loading."""

    @pytest.mark.asyncio
    async def test_ensure_loaded_imports_mlx_lm(self):
        """Verify _ensure_loaded calls mlx_lm.load."""
        provider = MLXProvider(model_path="test-model")
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("mlx_lm.load", return_value=(mock_model, mock_tokenizer)):
            await provider._ensure_loaded()

        assert provider._model is mock_model
        assert provider._tokenizer is mock_tokenizer

    @pytest.mark.asyncio
    async def test_ensure_loaded_skips_if_already_loaded(self, mlx_provider):
        """Should not reload if model is already set."""
        original_model = mlx_provider._model
        await mlx_provider._ensure_loaded()
        assert mlx_provider._model is original_model

    @pytest.mark.asyncio
    async def test_ensure_loaded_import_error(self):
        """Should raise ImportError if mlx_lm not installed."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "mlx_lm":
                raise ImportError("No module named 'mlx_lm'")
            return real_import(name, *args, **kwargs)

        provider = MLXProvider(model_path="test-model")

        with (
            patch("builtins.__import__", side_effect=mock_import),
            pytest.raises(ImportError, match="mlx-lm package is required"),
        ):
            await provider._ensure_loaded()

    @pytest.mark.asyncio
    async def test_adapter_path_passed(self):
        """Verify adapter_path is forwarded to mlx_lm.load."""
        provider = MLXProvider(model_path="test-model", adapter_path="/path/to/adapter")
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch(
            "mlx_lm.load", return_value=(mock_model, mock_tokenizer)
        ) as mock_load:
            await provider._ensure_loaded()

        mock_load.assert_called_once_with("test-model", adapter_path="/path/to/adapter")


class TestMLXProviderGenerateError:
    """Tests for MLX generation error handling."""

    @pytest.mark.asyncio
    async def test_generate_error_propagates(self, mlx_provider):
        """Generation errors should propagate."""
        with (
            patch("mlx_lm.generate", side_effect=RuntimeError("OOM")),
            pytest.raises(RuntimeError, match="OOM"),
        ):
            await mlx_provider.generate(
                messages=[{"role": "user", "content": "Hi"}],
            )
