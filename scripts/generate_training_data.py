#!/usr/bin/env python3
"""Generate synthetic SFT training pairs from OH documentation.

Uses the Groq API (Llama 3.3 70B) to generate realistic Q&A pairs
from the ingested OH documentation, with answers in the O-OH persona
defined by ``v1.txt``.

Output is Alpaca-style JSONL ready for :class:`FineTuner`.

Usage::

    python scripts/generate_training_data.py [--output data/training/oh_sft_v1.jsonl]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ZEROZERO = PROJECT_ROOT.parent

V1_PROMPT_PATH = ZEROZERO / "oh-companion-backend" / "prompts" / "v1.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "training" / "oh_sft_v1.jsonl"
PERSIST_DIR = PROJECT_ROOT / "data" / "oh_knowledge"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Prompt templates for Groq ────────────────────────────────────────

QUESTION_GEN_SYSTEM = """\
You generate realistic user questions that someone visiting the OH website might ask.
Given a documentation chunk, produce 2-3 natural questions a real person would ask.
Return ONLY a JSON array of strings — no explanation, no markdown fences."""

QUESTION_GEN_USER = """\
Documentation chunk:
---
{chunk}
---

Generate 2-3 natural user questions about this content. Return a JSON array of strings."""

ANSWER_GEN_USER = """\
You are answering as O-OH on the OH website. A visitor asks:

"{question}"

Relevant context from OH documentation:
---
{context}
---

Answer the question in character as O-OH. Be concise (1-3 sentences), warm, and helpful. \
No emojis. Always finish your sentences."""


class SyntheticDataGenerator:
    """Generate SFT training pairs from OH docs using Groq."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
        system_prompt: str = "",
        rate_limit_delay: float = 1.0,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model
        self.system_prompt = system_prompt
        self.rate_limit_delay = rate_limit_delay
        self._client = None

    def _get_client(self):
        """Lazy-init Groq async client."""
        if self._client is None:
            from groq import AsyncGroq

            if not self.api_key:
                raise ValueError("GROQ_API_KEY is required")
            self._client = AsyncGroq(api_key=self.api_key, timeout=60.0)
        return self._client

    async def _call_groq(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Make a single Groq API call with retry logic."""
        client = self._get_client()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 2
                    logger.warning("Groq call failed (%s), retrying in %ds…", e, wait)
                    await asyncio.sleep(wait)
                else:
                    raise
        return ""  # unreachable

    async def generate_questions(self, chunk: str) -> list[str]:
        """Generate 2-3 user questions from a documentation chunk."""
        messages = [
            {"role": "system", "content": QUESTION_GEN_SYSTEM},
            {"role": "user", "content": QUESTION_GEN_USER.format(chunk=chunk)},
        ]
        raw = await self._call_groq(messages, temperature=0.8, max_tokens=512)

        # Parse JSON array from response
        try:
            # Strip markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            questions = json.loads(cleaned)
            if isinstance(questions, list):
                return [q for q in questions if isinstance(q, str) and q.strip()]
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse questions from: %.100s…", raw)
        return []

    async def generate_answer(self, question: str, context: str) -> str:
        """Generate an O-OH persona answer for a question."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": ANSWER_GEN_USER.format(
                    question=question, context=context
                ),
            },
        ]
        return await self._call_groq(messages, temperature=0.6, max_tokens=300)

    async def generate_pairs_from_chunks(
        self,
        chunks: list[dict[str, str]],
    ) -> list[dict]:
        """Generate SFT pairs from a list of chunk dicts.

        Each chunk dict has ``text`` and ``source`` keys.

        Returns Alpaca-style dicts: {instruction, input, output, history}.
        """
        pairs: list[dict] = []
        total = len(chunks)

        for i, chunk in enumerate(chunks, 1):
            text = chunk["text"]
            source = chunk.get("source", "")

            logger.info("[%d/%d] Generating questions for: %s", i, total, source[:60])

            # Step 1: generate questions
            questions = await self.generate_questions(text)
            if not questions:
                logger.warning("  No questions generated, skipping chunk")
                continue

            await asyncio.sleep(self.rate_limit_delay)

            # Step 2: generate answers for each question
            for q in questions:
                answer = await self.generate_answer(q, text)
                if not answer.strip():
                    continue

                pairs.append({
                    "instruction": self.system_prompt,
                    "input": q,
                    "output": answer.strip(),
                    "history": [],
                })
                logger.info("  Q: %.70s…", q)
                logger.info("  A: %.70s…", answer.strip())

                await asyncio.sleep(self.rate_limit_delay)

            logger.info("  → %d pairs so far", len(pairs))

        return pairs


def _get_chunks_from_store() -> list[dict[str, str]]:
    """Read unique chunks from the ChromaDB store."""
    import chromadb

    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    collection = client.get_collection("oh_knowledge")

    # Get all documents
    results = collection.get(include=["documents", "metadatas"])
    chunks: list[dict[str, str]] = []
    seen_texts: set[str] = set()

    docs = results.get("documents") or []
    metas = results.get("metadatas") or []

    for doc, meta in zip(docs, metas):
        # Skip very short chunks and duplicates
        if not doc or len(doc.strip()) < 50:
            continue
        # Dedup by first 200 chars
        key = doc[:200]
        if key in seen_texts:
            continue
        seen_texts.add(key)

        chunks.append({
            "text": doc,
            "source": (meta or {}).get("source", ""),
        })

    return chunks


def save_pairs(pairs: list[dict], output_path: Path) -> None:
    """Write Alpaca-style JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    logger.info("Saved %d training pairs to %s", len(pairs), output_path)


async def run(output_path: Path) -> int:
    """Main generation pipeline."""
    # Load v1 system prompt
    if not V1_PROMPT_PATH.exists():
        logger.error("v1.txt not found at %s", V1_PROMPT_PATH)
        sys.exit(1)
    system_prompt = V1_PROMPT_PATH.read_text(encoding="utf-8").strip()
    logger.info("Loaded v1 system prompt (%d chars)", len(system_prompt))

    # Load .env if present
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    # Get chunks from store
    chunks = _get_chunks_from_store()
    logger.info("Loaded %d unique chunks from vector store", len(chunks))

    if not chunks:
        logger.error("No chunks in store — run ingest_oh_docs.py first")
        return 0

    # Generate pairs
    generator = SyntheticDataGenerator(
        system_prompt=system_prompt,
        rate_limit_delay=1.5,  # Groq free tier is ~30 req/min
    )

    start = time.time()
    pairs = await generator.generate_pairs_from_chunks(chunks)
    elapsed = time.time() - start

    logger.info(
        "Generated %d pairs from %d chunks in %.1f minutes",
        len(pairs),
        len(chunks),
        elapsed / 60,
    )

    if pairs:
        save_pairs(pairs, output_path)
    else:
        logger.error("No training pairs generated!")

    return len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic SFT training data from OH docs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL path (default: data/training/oh_sft_v1.jsonl)",
    )
    args = parser.parse_args()

    count = asyncio.run(run(args.output))
    if count == 0:
        sys.exit(1)
    logger.info("Done ✓ — %d training pairs ready", count)


if __name__ == "__main__":
    main()
