"""Retrieval-Augmented Generation - embeddings, vector store, retriever."""

from __future__ import annotations

from typing import TYPE_CHECKING

# Chunker is pure-Python and always importable.
from self_improving_agent.retrieval.chunker import Chunk, RecursiveChunker

# Embeddings / store / retriever depend on numpy, chromadb, and
# sentence-transformers which live in the optional [rag] extras.
# Import them only at type-check time so the package stays importable
# in CI environments that only install [dev].
if TYPE_CHECKING:
    from self_improving_agent.retrieval.embeddings import EmbeddingModel
    from self_improving_agent.retrieval.retriever import Retriever
    from self_improving_agent.retrieval.store import SearchResult, VectorStore

__all__ = [
    "Chunk",
    "EmbeddingModel",
    "RecursiveChunker",
    "Retriever",
    "SearchResult",
    "VectorStore",
]


def __getattr__(name: str) -> object:
    """Lazy-import RAG components that require optional deps."""
    if name == "EmbeddingModel":
        from self_improving_agent.retrieval.embeddings import EmbeddingModel

        return EmbeddingModel
    if name == "VectorStore":
        from self_improving_agent.retrieval.store import VectorStore

        return VectorStore
    if name == "SearchResult":
        from self_improving_agent.retrieval.store import SearchResult

        return SearchResult
    if name == "Retriever":
        from self_improving_agent.retrieval.retriever import Retriever

        return Retriever
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
