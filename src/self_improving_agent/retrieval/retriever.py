"""RAG retriever - ties embeddings, vector store, and chunking together.

This is the main entry point for the retrieval layer.  Other modules
(the context builder, the middleware, etc.) call :meth:`Retriever.search`
to get relevant knowledge chunks for a user query.

Usage::

    retriever = Retriever()
    await retriever.ingest("docs/faq.md", faq_text)
    results = await retriever.search("How do I reset my password?")
"""

from __future__ import annotations

from pathlib import Path

from self_improving_agent.retrieval.chunker import RecursiveChunker
from self_improving_agent.retrieval.embeddings import EmbeddingModel
from self_improving_agent.retrieval.store import SearchResult, VectorStore


class Retriever:
    """End-to-end RAG retriever: ingest documents, search by query.

    Parameters
    ----------
    embedding_model:
        An :class:`EmbeddingModel` instance.  If ``None``, a default
        ``all-MiniLM-L6-v2`` model is created.
    vector_store:
        A :class:`VectorStore` instance.  If ``None``, an ephemeral
        in-memory ChromaDB store is created.
    chunker:
        A :class:`RecursiveChunker` instance.  If ``None``, defaults
        to 512-char chunks with 50-char overlap.
    min_score:
        Minimum similarity score (0-1) to include in results.
    """

    def __init__(
        self,
        *,
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
        chunker: RecursiveChunker | None = None,
        min_score: float = 0.3,
    ) -> None:
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or VectorStore()
        self.chunker = chunker or RecursiveChunker()
        self.min_score = min_score

    # ── Ingestion ─────────────────────────────────────────────────────

    async def ingest(
        self,
        source: str,
        text: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> int:
        """Chunk, embed, and store a document.

        Parameters
        ----------
        source:
            Identifier for the document (e.g., file path or URL).
        text:
            The full document text.
        metadata:
            Extra metadata to attach to every chunk.

        Returns
        -------
        int
            Number of chunks ingested.
        """
        chunks = self.chunker.chunk_text(text, source=source)
        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)

        texts = [c.text for c in chunks]
        embeddings = await self.embedding_model.embed(texts)
        embedding_lists = [emb.tolist() for emb in embeddings]

        return await self.vector_store.add_chunks(chunks, embedding_lists)

    async def ingest_file(
        self,
        path: str | Path,
        *,
        metadata: dict[str, str] | None = None,
    ) -> int:
        """Read a text file and ingest it."""
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        return await self.ingest(str(p), text, metadata=metadata)

    # ── Search ────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Find the most relevant chunks for *query*.

        Returns up to *top_k* results whose similarity score
        exceeds :attr:`min_score`.
        """
        query_embedding = await self.embedding_model.embed_query(query)
        results = await self.vector_store.query(
            query_embedding=query_embedding.tolist(),
            top_k=top_k,
        )
        return [r for r in results if r.score >= self.min_score]

    # ── Context formatting ────────────────────────────────────────────

    async def get_context(
        self,
        query: str,
        *,
        top_k: int = 3,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """Search and return results formatted as a context string.

        This is the convenience method designed to plug into
        ``build_llm_context()`` in the companion backend.
        """
        results = await self.search(query, top_k=top_k)
        if not results:
            return ""
        parts: list[str] = []
        for r in results:
            header = f"[Source: {r.source}]" if r.source else ""
            parts.append(f"{header}\n{r.text}".strip())
        return separator.join(parts)

    # ── Management ────────────────────────────────────────────────────

    async def count(self) -> int:
        """Number of chunks in the vector store."""
        return await self.vector_store.count()

    async def reset(self) -> None:
        """Delete all stored chunks."""
        await self.vector_store.reset()
