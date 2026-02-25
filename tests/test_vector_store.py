"""Tests for VectorStore (ChromaDB wrapper)."""

from __future__ import annotations

import pytest

pytest.importorskip("chromadb", reason="chromadb required for vector store tests")
pytest.importorskip("numpy", reason="numpy required for vector store tests")

from self_improving_agent.retrieval.chunker import Chunk
from self_improving_agent.retrieval.store import SearchResult, VectorStore

# ── SearchResult ──────────────────────────────────────────────────────


class TestSearchResult:
    def test_defaults(self) -> None:
        r = SearchResult(text="hello", score=0.9)
        assert r.source == ""
        assert r.metadata == {}


# ── VectorStore ───────────────────────────────────────────────────────


class TestVectorStore:
    async def test_add_and_count(self) -> None:
        store = VectorStore(collection_name="test_add")
        chunks = [
            Chunk(text="first doc", source="a.md", chunk_index=0),
            Chunk(text="second doc", source="b.md", chunk_index=0),
        ]
        # Use simple embeddings (3-dim) for testing
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        added = await store.add_chunks(chunks, embeddings)
        assert added == 2
        assert await store.count() == 2

    async def test_add_empty(self) -> None:
        store = VectorStore(collection_name="test_empty")
        assert await store.add_chunks([], []) == 0

    async def test_query(self) -> None:
        store = VectorStore(collection_name="test_query")
        chunks = [
            Chunk(text="about cats", source="cats.md", chunk_index=0),
            Chunk(text="about dogs", source="dogs.md", chunk_index=0),
        ]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        await store.add_chunks(chunks, embeddings)

        # Query close to first embedding
        results = await store.query([0.9, 0.1, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].text == "about cats"
        assert results[0].score > results[1].score

    async def test_query_empty_store(self) -> None:
        store = VectorStore(collection_name="test_query_empty")
        # Ensure collection exists
        store._ensure_client()
        results = await store.query([1.0, 0.0, 0.0], top_k=5)
        assert results == []

    async def test_query_with_metadata(self) -> None:
        store = VectorStore(collection_name="test_meta")
        chunks = [
            Chunk(text="doc A", source="a.md", chunk_index=0),
        ]
        embeddings = [[1.0, 0.0, 0.0]]
        await store.add_chunks(chunks, embeddings)

        results = await store.query([1.0, 0.0, 0.0], top_k=1)
        assert results[0].source == "a.md"

    async def test_reset(self) -> None:
        store = VectorStore(collection_name="test_reset")
        chunks = [Chunk(text="temp", source="x", chunk_index=0)]
        await store.add_chunks(chunks, [[1.0, 0.0, 0.0]])
        assert await store.count() == 1

        await store.reset()
        assert await store.count() == 0

    async def test_upsert_deduplicates(self) -> None:
        store = VectorStore(collection_name="test_upsert")
        chunk = Chunk(text="doc", source="a.md", chunk_index=0)
        await store.add_chunks([chunk], [[1.0, 0.0, 0.0]])
        await store.add_chunks([chunk], [[1.0, 0.0, 0.0]])
        assert await store.count() == 1

    async def test_delete_collection(self) -> None:
        store = VectorStore(collection_name="test_delete")
        chunks = [Chunk(text="bye", source="x", chunk_index=0)]
        await store.add_chunks(chunks, [[1.0, 0.0, 0.0]])
        await store.delete_collection()
        assert store._collection is None
