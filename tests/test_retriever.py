"""Tests for Retriever (end-to-end RAG pipeline).

Uses mocked embeddings to avoid loading a real model in CI.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

np = pytest.importorskip("numpy", reason="numpy required for RAG tests")
pytest.importorskip("chromadb", reason="chromadb required for RAG tests")

from self_improving_agent.retrieval.retriever import Retriever  # noqa: E402
from self_improving_agent.retrieval.store import VectorStore  # noqa: E402


def _mock_embedding_model() -> AsyncMock:
    """Return a mock EmbeddingModel that produces deterministic embeddings."""
    model = AsyncMock()

    # embed() returns (N, 3) array - simple hash-based embeddings
    async def fake_embed(texts: list[str]) -> np.ndarray:
        result = []
        for t in texts:
            h = hash(t) % 1000
            result.append([h / 1000, (h + 333) % 1000 / 1000, (h + 666) % 1000 / 1000])
        return np.array(result, dtype=np.float32)

    async def fake_embed_query(query: str) -> np.ndarray:
        arr = await fake_embed([query])
        return arr[0]

    model.embed = fake_embed
    model.embed_query = fake_embed_query
    return model


class TestRetrieverIngest:
    async def test_ingest_returns_chunk_count(self) -> None:
        r = Retriever(
            embedding_model=_mock_embedding_model(),
            vector_store=VectorStore(collection_name="ingest_test"),
        )
        count = await r.ingest("doc.md", "Hello world. This is a test document.")
        assert count >= 1

    async def test_ingest_stores_chunks(self) -> None:
        r = Retriever(
            embedding_model=_mock_embedding_model(),
            vector_store=VectorStore(collection_name="ingest_stored"),
        )
        await r.ingest("doc.md", "Some text for testing.")
        assert await r.count() >= 1

    async def test_ingest_with_metadata(self) -> None:
        r = Retriever(
            embedding_model=_mock_embedding_model(),
            vector_store=VectorStore(collection_name="ingest_meta"),
        )
        count = await r.ingest(
            "doc.md",
            "Text with metadata.",
            metadata={"category": "faq"},
        )
        assert count >= 1

    async def test_ingest_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("File content for ingestion.")
        r = Retriever(
            embedding_model=_mock_embedding_model(),
            vector_store=VectorStore(collection_name="ingest_file"),
        )
        count = await r.ingest_file(f)
        assert count >= 1

    async def test_ingest_empty_text(self) -> None:
        r = Retriever(
            embedding_model=_mock_embedding_model(),
            vector_store=VectorStore(collection_name="ingest_empty"),
        )
        count = await r.ingest("empty.md", "")
        assert count == 0


class TestRetrieverSearch:
    async def test_search_returns_results(self) -> None:
        r = Retriever(
            embedding_model=_mock_embedding_model(),
            vector_store=VectorStore(collection_name="search_test"),
            min_score=0.0,  # accept everything for test
        )
        await r.ingest("doc.md", "Cats are great pets. Dogs are loyal friends.")
        results = await r.search("cats", top_k=5)
        assert len(results) >= 1

    async def test_search_empty_store(self) -> None:
        r = Retriever(
            embedding_model=_mock_embedding_model(),
            vector_store=VectorStore(collection_name="search_empty"),
            min_score=0.0,
        )
        results = await r.search("anything")
        assert results == []

    async def test_min_score_filtering(self) -> None:
        r = Retriever(
            embedding_model=_mock_embedding_model(),
            vector_store=VectorStore(collection_name="search_filter"),
            min_score=0.99,  # very high threshold
        )
        await r.ingest("doc.md", "Some content here.")
        results = await r.search("unrelated query", top_k=5)
        # With mocked embeddings, most results won't pass 0.99
        # This just tests the filtering logic works
        assert all(r.score >= 0.99 for r in results)


class TestRetrieverContext:
    async def test_get_context_returns_string(self) -> None:
        r = Retriever(
            embedding_model=_mock_embedding_model(),
            vector_store=VectorStore(collection_name="context_test"),
            min_score=0.0,
        )
        await r.ingest("faq.md", "How to reset password: go to settings.")
        ctx = await r.get_context("reset password")
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    async def test_get_context_empty(self) -> None:
        r = Retriever(
            embedding_model=_mock_embedding_model(),
            vector_store=VectorStore(collection_name="context_empty"),
        )
        ctx = await r.get_context("anything")
        assert ctx == ""

    async def test_get_context_includes_source(self) -> None:
        r = Retriever(
            embedding_model=_mock_embedding_model(),
            vector_store=VectorStore(collection_name="context_source"),
            min_score=0.0,
        )
        await r.ingest("faq.md", "Answer to FAQ question.")
        ctx = await r.get_context("faq")
        assert "faq.md" in ctx


class TestRetrieverManagement:
    async def test_count(self) -> None:
        r = Retriever(
            embedding_model=_mock_embedding_model(),
            vector_store=VectorStore(collection_name="mgmt_count"),
        )
        assert await r.count() == 0
        await r.ingest("doc.md", "Some text.")
        assert await r.count() >= 1

    async def test_reset(self) -> None:
        r = Retriever(
            embedding_model=_mock_embedding_model(),
            vector_store=VectorStore(collection_name="mgmt_reset"),
        )
        await r.ingest("doc.md", "Some text.")
        assert await r.count() >= 1
        await r.reset()
        assert await r.count() == 0
