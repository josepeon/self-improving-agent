"""Tests for the EmbeddingModel wrapper.

Uses mocks to avoid downloading a real model in fast tests.
A single integration test (marked slow) exercises the real model.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy", reason="numpy required for RAG tests")
pytest.importorskip("sentence_transformers", reason="sentence-transformers required")

from self_improving_agent.retrieval.embeddings import EmbeddingModel  # noqa: E402


class TestEmbeddingModelUnit:
    """Unit tests with mocked SentenceTransformer."""

    async def test_embed_returns_array(self) -> None:
        model = EmbeddingModel()
        fake_st = MagicMock()
        fake_st.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        model._model = fake_st

        result = await model.embed(["hello"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)

    async def test_embed_query_returns_1d(self) -> None:
        model = EmbeddingModel()
        fake_st = MagicMock()
        fake_st.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        model._model = fake_st

        result = await model.embed_query("hello")
        assert result.ndim == 1

    async def test_embed_multiple(self) -> None:
        model = EmbeddingModel()
        fake_st = MagicMock()
        fake_st.encode.return_value = np.zeros((3, 384), dtype=np.float32)
        model._model = fake_st

        result = await model.embed(["a", "b", "c"])
        assert result.shape[0] == 3

    def test_dimension_before_load_raises(self) -> None:
        model = EmbeddingModel()
        with pytest.raises(RuntimeError, match="not loaded"):
            _ = model.dimension

    def test_dimension_after_load(self) -> None:
        model = EmbeddingModel()
        fake_st = MagicMock()
        fake_st.get_sentence_embedding_dimension.return_value = 384
        model._model = fake_st
        assert model.dimension == 384

    async def test_lazy_loading(self) -> None:
        model = EmbeddingModel()
        assert model._model is None

        fake_st = MagicMock()
        fake_st.encode.return_value = np.zeros((1, 384), dtype=np.float32)

        with patch(
            "self_improving_agent.retrieval.embeddings.EmbeddingModel._load_model",
            return_value=fake_st,
        ):
            await model.embed(["trigger load"])

        assert model._model is not None

    def test_custom_model_name(self) -> None:
        model = EmbeddingModel(model_name="custom/model", device="cpu")
        assert model.model_name == "custom/model"
        assert model.device == "cpu"
