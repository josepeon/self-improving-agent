"""Vector store abstraction backed by ChromaDB.

Provides async methods for adding, querying, and managing document
collections used by the RAG retriever.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from self_improving_agent.retrieval.chunker import Chunk


@dataclass
class SearchResult:
    """A single result from a vector similarity search."""

    text: str
    score: float  # similarity (higher = more relevant)
    source: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


class VectorStore:
    """Async wrapper around a ChromaDB collection.

    Parameters
    ----------
    collection_name:
        Name of the ChromaDB collection.
    persist_directory:
        If provided, ChromaDB persists to disk at this path.
        If ``None``, uses an ephemeral in-memory store.
    """

    def __init__(
        self,
        collection_name: str = "oh_knowledge",
        persist_directory: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client: Any | None = None
        self._collection: Any | None = None

    def _ensure_client(self) -> None:
        """Lazily initialise the ChromaDB client and collection."""
        if self._client is not None:
            return

        import chromadb

        if self.persist_directory:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Writes ────────────────────────────────────────────────────────

    async def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """Add chunked documents with pre-computed embeddings.

        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        ids = [f"{c.source}:{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {"source": c.source, "chunk_index": str(c.chunk_index), **c.metadata}
            for c in chunks
        ]

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            partial(
                self._add,
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            ),
        )
        return len(chunks)

    def _add(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> None:
        self._ensure_client()
        assert self._collection is not None
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # ── Reads ─────────────────────────────────────────────────────────

    async def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Find the *top_k* most similar chunks to *query_embedding*."""
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(
            None,
            partial(
                self._query, query_embedding=query_embedding, top_k=top_k, where=where
            ),
        )
        return raw

    def _query(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        where: dict[str, Any] | None,
    ) -> list[SearchResult]:
        self._ensure_client()
        assert self._collection is not None

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        search_results: list[SearchResult] = []
        if results and results["ids"] and results["ids"][0]:
            docs = results["documents"][0] if results["documents"] else []
            distances = results["distances"][0] if results["distances"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []

            for i, doc in enumerate(docs):
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity score: 1 - (distance / 2)
                distance = distances[i] if i < len(distances) else 0.0
                similarity = 1.0 - (distance / 2.0)
                meta = metadatas[i] if i < len(metadatas) else {}
                search_results.append(
                    SearchResult(
                        text=doc,
                        score=similarity,
                        source=meta.get("source", ""),
                        metadata=meta,
                    )
                )

        return search_results

    # ── Management ────────────────────────────────────────────────────

    async def count(self) -> int:
        """Return the number of documents in the collection."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._count)

    def _count(self) -> int:
        self._ensure_client()
        assert self._collection is not None
        return self._collection.count()

    async def delete_collection(self) -> None:
        """Delete the entire collection."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._delete_collection)

    def _delete_collection(self) -> None:
        self._ensure_client()
        assert self._client is not None
        self._client.delete_collection(self.collection_name)
        self._collection = None

    async def reset(self) -> None:
        """Delete and re-create the collection."""
        await self.delete_collection()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._recreate)

    def _recreate(self) -> None:
        self._ensure_client()
        assert self._client is not None
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
