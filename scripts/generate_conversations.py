#!/usr/bin/env python3
"""Generate multi-turn synthetic conversations using the local OH model.

The local v1.0-oh model plays O-OH while Groq (70B) simulates diverse
users *and* judges conversation quality.  Only conversations scoring
above the quality threshold are kept for training.

Pipeline::

    1. Sample doc chunk → Groq generates a user persona + opening message
    2. Local model (O-OH) responds
    3. Groq generates a follow-up as the user
    4. Repeat for 3-5 turns
    5. Groq judges the full conversation (1-10 score)
    6. Keep conversations scoring ≥ 7

Output is Alpaca-style JSONL with multi-turn ``history`` ready for
:class:`FineTuner`.

Usage::

    python scripts/generate_conversations.py [--rounds 200] [--resume]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

from self_improving_agent.providers.groq_models import request_opts, resolve_groq_model

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ZEROZERO = PROJECT_ROOT.parent

V1_PROMPT_PATH = ZEROZERO / "oh-companion-backend" / "prompts" / "v1.txt"
ADAPTER_PATH = PROJECT_ROOT / "models" / "finetuned" / "v1.1-oh" / "adapters"
BASE_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "training" / "oh_conversations_v2.jsonl"
PERSIST_DIR = PROJECT_ROOT / "data" / "oh_knowledge"

JUDGE_THRESHOLD = 7  # minimum score to keep a conversation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Prompt templates ─────────────────────────────────────────────────

USER_PERSONAS = [
    "a curious first-time visitor who just landed on the OH website",
    "a game developer interested in integrating commerce into their 3D world",
    "a startup founder exploring spatial commerce for their platform",
    "a skeptical investor evaluating OH as a potential investment",
    "a brand manager wanting to sell products in virtual environments",
    "a technical architect evaluating OH's API and infrastructure",
    "a journalist writing an article about the future of spatial commerce",
    "a teenager who stumbled onto the site and is confused about what OH does",
    "a Shopify merchant curious about expanding into spatial experiences",
    "a casual browser who just wants to chat and see how the AI works",
    "a developer who wants to understand how O-OH agents are built",
    "a partner at a gaming studio interested in the Commerce Runtime",
]

USER_SIM_SYSTEM = """\
You simulate a realistic website visitor having a conversation with an AI assistant.
You are {persona}.
Given the conversation so far and some context about OH, generate your next message.

Rules:
- Stay in character as the visitor persona
- Be natural — ask follow-ups, express curiosity or skepticism
- Keep messages 1-3 sentences
- Sometimes go off-topic slightly, like a real person would
- If the assistant gave a good answer, dig deeper or ask about something related
- Return ONLY your message text, nothing else"""

USER_SIM_FIRST = """\
You are visiting the OH website for the first time.

Context about OH (from their docs):
---
{context}
---

Generate your opening message to the AI assistant on the site. Be natural and conversational.
Return ONLY the message text."""

USER_SIM_FOLLOWUP = """\
Conversation so far:
{history}

Context about OH:
---
{context}
---

Generate your next message as the visitor. Return ONLY the message text."""

JUDGE_SYSTEM = """\
You evaluate conversations between a website visitor and O-OH, an AI assistant \
for the OH spatial commerce company. Score the ASSISTANT's performance only.

Scoring criteria (1-10):
- **Accuracy**: Does the assistant give correct information about OH?
- **Persona**: Does it sound like O-OH (warm, concise, helpful, no emojis)?
- **Helpfulness**: Does it actually answer the visitor's questions?
- **Coherence**: Do responses make sense in context and flow naturally?
- **Brevity**: Are answers concise (1-3 sentences) like a voice assistant?

Return ONLY a JSON object: {"score": <int>, "reason": "<one sentence>"}"""

JUDGE_USER = """\
Conversation:
{conversation}

Rate the ASSISTANT's overall performance (1-10). \
Return ONLY: {{"score": <int>, "reason": "<one sentence>"}}"""


class ConversationGenerator:
    """Generate multi-turn conversations using local model + Groq."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        groq_model: str | None = None,
        system_prompt: str = "",
        adapter_path: str | Path = ADAPTER_PATH,
        base_model: str = BASE_MODEL,
        rate_limit_delay: float = 1.5,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.groq_model = groq_model or resolve_groq_model()
        self.system_prompt = system_prompt
        self.adapter_path = str(adapter_path)
        self.base_model = base_model
        self.rate_limit_delay = rate_limit_delay
        self._groq_client = None
        self._local_model = None
        self._local_tokenizer = None

    # ── Groq (user simulation + judge) ──────────────────────────────

    def _get_groq(self):
        if self._groq_client is None:
            from groq import AsyncGroq

            if not self.api_key:
                raise ValueError("GROQ_API_KEY is required")
            self._groq_client = AsyncGroq(api_key=self.api_key, timeout=60.0)
        return self._groq_client

    async def _call_groq(
        self, messages: list[dict], *, temperature: float = 0.7, max_tokens: int = 512
    ) -> str:
        client = self._get_groq()
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=self.groq_model,
                    messages=messages,
                    temperature=temperature,
                    **request_opts(self.groq_model, max_tokens),
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                if attempt < 2:
                    wait = (attempt + 1) * 3
                    logger.warning("Groq error (%s), retry in %ds…", e, wait)
                    await asyncio.sleep(wait)
                else:
                    raise
        return ""

    # ── Local model (O-OH) ──────────────────────────────────────────

    def _load_local_model(self):
        """Lazy-load the fine-tuned local model."""
        if self._local_model is None:
            from mlx_lm import load

            logger.info("Loading local model with adapter from %s", self.adapter_path)
            self._local_model, self._local_tokenizer = load(
                self.base_model,
                adapter_path=self.adapter_path,
            )
        return self._local_model, self._local_tokenizer

    def _generate_local(self, messages: list[dict], *, max_tokens: int = 200) -> str:
        """Generate a response from the local fine-tuned model."""
        from mlx_lm import generate

        model, tokenizer = self._load_local_model()
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens)

    # ── Conversation generation ─────────────────────────────────────

    async def generate_user_opening(self, persona: str, context: str) -> str:
        """Groq generates the visitor's first message."""
        messages = [
            {
                "role": "system",
                "content": USER_SIM_SYSTEM.format(persona=persona),
            },
            {"role": "user", "content": USER_SIM_FIRST.format(context=context)},
        ]
        return (
            await self._call_groq(messages, temperature=0.9, max_tokens=150)
        ).strip()

    async def generate_user_followup(
        self, persona: str, history_text: str, context: str
    ) -> str:
        """Groq generates a follow-up visitor message."""
        messages = [
            {
                "role": "system",
                "content": USER_SIM_SYSTEM.format(persona=persona),
            },
            {
                "role": "user",
                "content": USER_SIM_FOLLOWUP.format(
                    history=history_text, context=context
                ),
            },
        ]
        return (
            await self._call_groq(messages, temperature=0.8, max_tokens=150)
        ).strip()

    def generate_oh_response(self, conversation: list[dict]) -> str:
        """Local model generates O-OH's response."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(conversation)
        return self._generate_local(messages, max_tokens=200).strip()

    async def judge_conversation(self, conversation: list[dict]) -> tuple[int, str]:
        """Groq judges conversation quality, returns (score, reason)."""
        # Format conversation for the judge
        lines = []
        for msg in conversation:
            role = msg["role"].upper()
            lines.append(f"{role}: {msg['content']}")
        conv_text = "\n".join(lines)

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(conversation=conv_text)},
        ]
        raw = await self._call_groq(messages, temperature=0.3, max_tokens=100)

        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(cleaned)
            return int(result.get("score", 0)), result.get("reason", "")
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Failed to parse judge output: %.100s", raw)
            return 0, "parse_error"

    async def generate_conversation(
        self, context: str, *, min_turns: int = 3, max_turns: int = 5
    ) -> dict | None:
        """Generate a full multi-turn conversation and judge it.

        Returns an Alpaca-style dict if quality passes, else None.
        """
        persona = random.choice(USER_PERSONAS)
        num_turns = random.randint(min_turns, max_turns)
        conversation: list[dict] = []

        # Turn 1: user opens
        user_msg = await self.generate_user_opening(persona, context)
        if not user_msg:
            return None
        conversation.append({"role": "user", "content": user_msg})

        # Turn 1: O-OH responds (local model, sync — run in executor)
        loop = asyncio.get_running_loop()
        oh_msg = await loop.run_in_executor(
            None, self.generate_oh_response, list(conversation)
        )
        if not oh_msg:
            return None
        conversation.append({"role": "assistant", "content": oh_msg})

        # Remaining turns
        for _turn in range(num_turns - 1):
            await asyncio.sleep(self.rate_limit_delay)

            # Format history for the user simulator
            history_text = "\n".join(
                f"{'User' if m['role'] == 'user' else 'O-OH'}: {m['content']}"
                for m in conversation
            )

            # User follow-up (Groq)
            user_msg = await self.generate_user_followup(persona, history_text, context)
            if not user_msg:
                break
            conversation.append({"role": "user", "content": user_msg})

            # O-OH responds (local model)
            oh_msg = await loop.run_in_executor(
                None, self.generate_oh_response, list(conversation)
            )
            if not oh_msg:
                break
            conversation.append({"role": "assistant", "content": oh_msg})

        # Judge the conversation
        await asyncio.sleep(self.rate_limit_delay)
        score, reason = await self.judge_conversation(conversation)

        logger.info(
            "  Score: %d/10 (%s) — %d turns, persona: %s",
            score,
            reason[:60],
            len(conversation) // 2,
            persona[:40],
        )

        if score < JUDGE_THRESHOLD:
            return None

        # Convert to Alpaca-style with history
        # We store the full conversation: system prompt as instruction,
        # last user message as input, last assistant message as output,
        # earlier turns as history
        history = conversation[:-2]  # all but last exchange
        last_user = conversation[-2]["content"]
        last_assistant = conversation[-1]["content"]

        return {
            "instruction": self.system_prompt,
            "input": last_user,
            "output": last_assistant,
            "history": history,
            "metadata": {
                "persona": persona,
                "score": score,
                "reason": reason,
                "num_turns": len(conversation) // 2,
            },
        }


# ── Chunk loader ─────────────────────────────────────────────────────


def _get_chunks_from_store() -> list[dict[str, str]]:
    """Load all chunks from ChromaDB."""
    import chromadb

    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    collection = client.get_collection("oh_knowledge")
    results = collection.get(include=["documents", "metadatas"])
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []

    chunks = []
    seen: set[str] = set()
    for doc, meta in zip(docs, metas, strict=False):
        if not doc or len(doc.strip()) < 50:
            continue
        key = doc[:200]
        if key in seen:
            continue
        seen.add(key)
        chunks.append({"text": doc, "source": (meta or {}).get("source", "")})

    logger.info("Loaded %d unique chunks from vector store", len(chunks))
    return chunks


def _count_existing(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


# ── Main pipeline ────────────────────────────────────────────────────


async def run(
    output_path: Path,
    *,
    rounds: int = 200,
    resume: bool = False,
    adapter_path: Path = ADAPTER_PATH,
) -> int:
    """Generate conversations, judge them, keep the good ones."""
    # Load system prompt
    if not V1_PROMPT_PATH.exists():
        logger.error("v1.txt not found at %s", V1_PROMPT_PATH)
        sys.exit(1)
    system_prompt = V1_PROMPT_PATH.read_text(encoding="utf-8").strip()
    logger.info("Loaded v1 system prompt (%d chars)", len(system_prompt))

    # Load .env
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    # Check adapter exists
    if not adapter_path.exists():
        logger.error(
            "Adapter not found at %s — run train_oh_model.py first", adapter_path
        )
        sys.exit(1)

    # Load chunks
    chunks = _get_chunks_from_store()
    if not chunks:
        logger.error("No chunks — run ingest_oh_docs.py first")
        sys.exit(1)

    # Resume support
    start_round = 0
    if resume:
        existing = _count_existing(output_path)
        if existing > 0:
            start_round = existing
            logger.info(
                "Resuming from round %d (%d existing conversations)",
                start_round,
                existing,
            )
    elif output_path.exists():
        output_path.unlink()

    # Generator
    generator = ConversationGenerator(
        system_prompt=system_prompt,
        adapter_path=adapter_path,
        rate_limit_delay=1.5,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    rejected = 0
    start = time.time()

    with open(output_path, "a") as outfile:
        for i in range(start_round, rounds):
            chunk = random.choice(chunks)
            logger.info(
                "[%d/%d] Generating conversation from: %s",
                i + 1,
                rounds,
                chunk["source"][:50],
            )

            try:
                result = await generator.generate_conversation(chunk["text"])
            except Exception as e:
                logger.error("  Error: %s", e)
                await asyncio.sleep(3)
                continue

            if result:
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                outfile.flush()
                kept += 1
                logger.info("  ✓ Kept (total: %d kept, %d rejected)", kept, rejected)
            else:
                rejected += 1
                logger.info(
                    "  ✗ Rejected (total: %d kept, %d rejected)", kept, rejected
                )

    elapsed = time.time() - start
    total = _count_existing(output_path)
    logger.info(
        "Done: %d kept, %d rejected (%.0f%% pass rate) in %.1f minutes",
        kept,
        rejected,
        (kept / max(1, kept + rejected)) * 100,
        elapsed / 60,
    )
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multi-turn conversations with Groq-as-judge"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=200,
        help="Number of conversations to attempt (default: 200)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=ADAPTER_PATH,
        help=f"Path to adapter directory (default: {ADAPTER_PATH})",
    )
    args = parser.parse_args()

    count = asyncio.run(
        run(
            args.output,
            rounds=args.rounds,
            resume=args.resume,
            adapter_path=args.adapter,
        )
    )
    if count == 0:
        sys.exit(1)
    logger.info("Done ✓ — %d quality conversations ready for training", count)


if __name__ == "__main__":
    main()
