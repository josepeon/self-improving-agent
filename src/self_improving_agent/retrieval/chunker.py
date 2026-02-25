"""Document chunking strategies for RAG ingestion.

Splits text into overlapping chunks suitable for embedding and retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class Chunk:
    """A single text chunk with provenance metadata."""

    text: str
    source: str = ""
    chunk_index: int = 0
    metadata: dict[str, str] = field(default_factory=dict)


class RecursiveChunker:
    """Split text recursively by a hierarchy of separators.

    Tries to split on paragraph boundaries first, then sentences,
    then words, to keep semantically coherent pieces together.

    Parameters
    ----------
    chunk_size:
        Target maximum characters per chunk.
    chunk_overlap:
        Number of overlapping characters between consecutive chunks.
    separators:
        Ordered list of separator strings to try.
    """

    DEFAULT_SEPARATORS: ClassVar[list[str]] = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            msg = "chunk_overlap must be less than chunk_size"
            raise ValueError(msg)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk_text(self, text: str, *, source: str = "") -> list[Chunk]:
        """Split *text* into :class:`Chunk` objects."""
        pieces = self._split(text, self.separators)
        merged = self._merge(pieces)
        return [
            Chunk(text=t, source=source, chunk_index=i) for i, t in enumerate(merged)
        ]

    # ── Private helpers ───────────────────────────────────────────────

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split *text* using the first workable separator."""
        if not text:
            return []

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else []

        if sep == "":
            # Character-level fallback
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        parts = text.split(sep)
        good: list[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    good.append(current)
                # If single part exceeds chunk_size, recurse with finer separator
                if len(part) > self.chunk_size and remaining_seps:
                    good.extend(self._split(part, remaining_seps))
                else:
                    current = part

        if current:
            good.append(current)
        return good

    def _merge(self, pieces: list[str]) -> list[str]:
        """Merge pieces with overlap to ensure context continuity."""
        if not pieces:
            return []
        if self.chunk_overlap == 0:
            return pieces

        merged: list[str] = [pieces[0]]
        for piece in pieces[1:]:
            prev = merged[-1]
            # Take overlap from end of previous chunk
            overlap_text = (
                prev[-self.chunk_overlap :] if len(prev) > self.chunk_overlap else prev
            )
            combined = overlap_text + piece
            merged.append(combined)
        return merged
