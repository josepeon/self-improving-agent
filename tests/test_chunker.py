"""Tests for the document chunker."""

from __future__ import annotations

import pytest

from self_improving_agent.retrieval.chunker import Chunk, RecursiveChunker

# ── Chunk dataclass ───────────────────────────────────────────────────


class TestChunkDataclass:
    def test_defaults(self) -> None:
        c = Chunk(text="hello")
        assert c.source == ""
        assert c.chunk_index == 0
        assert c.metadata == {}

    def test_custom_fields(self) -> None:
        c = Chunk(text="x", source="doc.md", chunk_index=3, metadata={"k": "v"})
        assert c.source == "doc.md"
        assert c.chunk_index == 3
        assert c.metadata["k"] == "v"


# ── RecursiveChunker ──────────────────────────────────────────────────


class TestRecursiveChunker:
    def test_short_text_single_chunk(self) -> None:
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk_text("Hello world", source="test")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"
        assert chunks[0].source == "test"
        assert chunks[0].chunk_index == 0

    def test_splits_on_paragraph(self) -> None:
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=0)
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 2
        assert all(len(c.text) <= 30 for c in chunks)

    def test_splits_on_newline(self) -> None:
        text = "Line one.\nLine two.\nLine three."
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 2

    def test_splits_on_sentence(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        chunker = RecursiveChunker(chunk_size=25, chunk_overlap=0)
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 2

    def test_overlap_adds_context(self) -> None:
        text = "AAAA\n\nBBBB\n\nCCCC"
        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=3)
        chunks = chunker.chunk_text(text)
        # Second chunk should contain overlap from first
        if len(chunks) > 1:
            assert len(chunks[1].text) > len("BBBB")

    def test_chunk_indices_sequential(self) -> None:
        text = "A\n\nB\n\nC\n\nD"
        chunker = RecursiveChunker(chunk_size=5, chunk_overlap=0)
        chunks = chunker.chunk_text(text)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_empty_text(self) -> None:
        chunker = RecursiveChunker()
        assert chunker.chunk_text("") == []

    def test_invalid_overlap(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            RecursiveChunker(chunk_size=10, chunk_overlap=10)

    def test_source_propagated(self) -> None:
        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=0)
        chunks = chunker.chunk_text("Short text here now", source="faq.md")
        for c in chunks:
            assert c.source == "faq.md"

    def test_custom_separators(self) -> None:
        text = "a|b|c|d"
        chunker = RecursiveChunker(chunk_size=5, chunk_overlap=0, separators=["|", ""])
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 2
