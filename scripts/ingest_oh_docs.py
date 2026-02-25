#!/usr/bin/env python3
"""Ingest OH documentation into the RAG vector store.

Reads all Markdown files from oh-sss/ and the v1 persona prompt from
oh-companion-backend/prompts/, then chunks, embeds, and stores them
in a persistent ChromaDB at ``data/oh_knowledge/``.

Usage::

    python scripts/ingest_oh_docs.py [--reset]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Resolve project paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ZEROZERO = PROJECT_ROOT.parent  # …/ZEROZERO/

OH_SSS_DIR = ZEROZERO / "oh-sss"
PROMPTS_DIR = ZEROZERO / "oh-companion-backend" / "prompts"
PERSIST_DIR = PROJECT_ROOT / "data" / "oh_knowledge"

# Category metadata by directory
CATEGORY_MAP = {
    "projects": "project",
    "planning": "planning",
    "product": "product",
    "legal": "legal",
    "strategy": "strategy",
    "release": "release",
    "go-to-market": "go-to-market",
    "context": "context",
    "local": "context",
    "prompts": "persona",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


def discover_files() -> list[tuple[Path, dict[str, str]]]:
    """Find all OH docs and return (path, metadata) pairs."""
    files: list[tuple[Path, dict[str, str]]] = []

    # oh-sss/docs/**/*.md
    docs_dir = OH_SSS_DIR / "docs"
    if docs_dir.exists():
        for md in sorted(docs_dir.rglob("*.md")):
            category = _infer_category(md)
            files.append((md, {"source_repo": "oh-sss", "category": category}))

    # oh-sss/local/**/*.md
    local_dir = OH_SSS_DIR / "local"
    if local_dir.exists():
        for md in sorted(local_dir.rglob("*.md")):
            files.append((md, {"source_repo": "oh-sss", "category": "context"}))

    # oh-sss/README.md
    readme = OH_SSS_DIR / "README.md"
    if readme.exists():
        files.append((readme, {"source_repo": "oh-sss", "category": "overview"}))

    # oh-companion-backend/prompts/v1.txt (production persona)
    v1 = PROMPTS_DIR / "v1.txt"
    if v1.exists():
        files.append(
            (v1, {"source_repo": "oh-companion-backend", "category": "persona"})
        )

    return files


def _infer_category(path: Path) -> str:
    """Infer a category tag from the file's parent directory."""
    for part in path.parts:
        if part in CATEGORY_MAP:
            return CATEGORY_MAP[part]
    return "general"


async def ingest_all(*, reset: bool = False) -> int:
    """Ingest all discovered OH docs into the vector store."""
    # Import here so script startup is fast even without deps
    from self_improving_agent.retrieval import Retriever, VectorStore

    store = VectorStore(
        collection_name="oh_knowledge",
        persist_directory=str(PERSIST_DIR),
    )
    retriever = Retriever(vector_store=store)

    if reset:
        logger.info("Resetting existing vector store…")
        await retriever.reset()

    files = discover_files()
    if not files:
        logger.error("No OH documentation files found!")
        logger.error("Expected oh-sss at: %s", OH_SSS_DIR)
        logger.error("Expected prompts at: %s", PROMPTS_DIR)
        return 0

    logger.info("Found %d files to ingest", len(files))

    total_chunks = 0
    for path, metadata in files:
        try:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                logger.warning("Skipping empty file: %s", path.name)
                continue

            n = await retriever.ingest(
                source=str(path.relative_to(ZEROZERO)),
                text=text,
                metadata=metadata,
            )
            total_chunks += n
            logger.info("  %-50s → %3d chunks", path.name, n)
        except Exception:
            logger.exception("Failed to ingest %s", path)

    total_stored = await retriever.count()
    logger.info(
        "Ingestion complete: %d new chunks from %d files (%d total in store)",
        total_chunks,
        len(files),
        total_stored,
    )
    return total_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest OH docs into RAG store")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing store before ingesting",
    )
    args = parser.parse_args()

    chunks = asyncio.run(ingest_all(reset=args.reset))
    if chunks == 0:
        logger.error("No chunks ingested — check file paths above")
        sys.exit(1)
    logger.info("Done ✓")


if __name__ == "__main__":
    main()
