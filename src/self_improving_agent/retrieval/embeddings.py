"""Embedding model wrapper for RAG retrieval.

Provides a thin async interface over ``sentence-transformers`` so the
rest of the retrieval stack never touches the underlying library directly.

Default model: ``all-MiniLM-L6-v2`` (384-dim, ~80 MB, fast on CPU).
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Async wrapper around a SentenceTransformer model.

    The model is loaded lazily on first call to :meth:`embed`.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.
    device:
        ``"cpu"``, ``"mps"`` (Apple Silicon), or ``"cuda"``.
        Defaults to ``None`` (auto-detect).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        *,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(
                None,
                partial(self._load_model),
            )

    def _load_model(self) -> Any:
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {}
        if self.device is not None:
            kwargs["device"] = self.device
        return SentenceTransformer(self.model_name, **kwargs)

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Return embeddings for *texts* as a ``(N, dim)`` float32 array."""
        await self._ensure_loaded()
        loop = asyncio.get_running_loop()
        result: np.ndarray = await loop.run_in_executor(
            None,
            partial(self._model.encode, texts, normalize_embeddings=True),
        )
        return result

    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string, returning a 1-D array."""
        result = await self.embed([query])
        row: np.ndarray = result[0]
        return row

    @property
    def dimension(self) -> int:
        """Embedding dimensionality (available after first embed call)."""
        if self._model is None:
            msg = "Model not loaded yet; call embed() first"
            raise RuntimeError(msg)
        return int(self._model.get_sentence_embedding_dimension())
