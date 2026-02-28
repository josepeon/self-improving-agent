#!/usr/bin/env python3
"""Convert synced companion logs into training-ready JSONL.

Reads from ``data/companion_logs/*.jsonl`` (output of ``sync_logs.py``),
groups messages into multi-turn conversations by proximity in time,
optionally scores quality via Groq, and outputs Alpaca-style JSONL
ready for the training pipeline.

Usage::

    # Basic — convert all synced logs
    python scripts/prepare_real_data.py

    # Score quality with Groq (slower, more selective)
    python scripts/prepare_real_data.py --score

    # Custom paths
    python scripts/prepare_real_data.py --input-dir data/companion_logs --output data/training/real_conversations.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ZEROZERO = PROJECT_ROOT.parent

V1_PROMPT_PATH = ZEROZERO / "oh-companion-backend" / "prompts" / "v1.txt"

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "companion_logs"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "training" / "real_conversations.jsonl"

# Minimum message length to consider a real interaction
MIN_USER_MSG_LENGTH = 5
MIN_ASSISTANT_MSG_LENGTH = 10

# Maximum gap between messages to consider them part of the same session
SESSION_GAP_SECONDS = 300  # 5 minutes

# Minimum quality score from Groq judge to keep a conversation
JUDGE_THRESHOLD = 7

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Groq judge (reuses same scoring as generate_conversations.py) ────


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


async def score_conversation(
    conversation: list[dict],
    groq_client: object,
    groq_model: str = "llama-3.3-70b-versatile",
) -> tuple[int, str]:
    """Score a conversation using Groq as judge.

    Returns (score, reason).
    """
    lines = []
    for msg in conversation:
        role = msg["role"].upper()
        lines.append(f"{role}: {msg['content']}")
    conv_text = "\n".join(lines)

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER.format(conversation=conv_text)},
    ]

    try:
        resp = await groq_client.chat.completions.create(
            model=groq_model,
            messages=messages,
            temperature=0.3,
            max_tokens=100,
        )
        raw = resp.choices[0].message.content or ""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(cleaned)
        return int(result.get("score", 0)), result.get("reason", "")
    except Exception as e:
        logger.warning("Judge failed: %s", e)
        return 0, f"error: {e}"


# ── Data loading and grouping ────────────────────────────────────────


def load_all_logs(input_dir: Path) -> list[dict]:
    """Load all JSONL entries from the companion_logs directory.

    Args:
        input_dir: Directory containing date-partitioned JSONL files.

    Returns:
        All log entries sorted by timestamp.
    """
    entries = []

    if not input_dir.exists():
        logger.warning("Input directory not found: %s", input_dir)
        return entries

    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue

    # Sort by timestamp
    entries.sort(key=lambda e: e.get("timestamp", ""))
    logger.info(
        "Loaded %d log entries from %s",
        len(entries),
        input_dir,
    )
    return entries


def is_real_interaction(entry: dict) -> bool:
    """Check if a log entry represents a genuine user interaction.

    Filters out:
    - Internal system calls (fact-extraction, summarization)
    - Very short messages (likely noise)
    - Missing fields
    """
    user_msg = entry.get("user_message", "")
    assistant_msg = entry.get("assistant_response", "")

    # Must have both user message and response
    if not user_msg or not assistant_msg:
        return False

    # Filter internal system calls
    internal_prefixes = (
        "Analyze this conversation exchange",
        "Extract ",
        "Summarize this conversation",
    )
    if any(user_msg.startswith(prefix) for prefix in internal_prefixes):
        return False

    # Minimum length checks
    if len(user_msg.strip()) < MIN_USER_MSG_LENGTH:
        return False
    return len(assistant_msg.strip()) >= MIN_ASSISTANT_MSG_LENGTH


def group_into_sessions(entries: list[dict]) -> list[list[dict]]:
    """Group log entries into multi-turn conversation sessions.

    Uses ``history_length`` field and timestamp proximity to determine
    session boundaries.  A new session starts when:
    - ``history_length`` is 0 (fresh conversation)
    - More than ``SESSION_GAP_SECONDS`` elapsed since the last message

    Args:
        entries: Filtered, sorted log entries.

    Returns:
        List of sessions, where each session is a list of entries.
    """
    if not entries:
        return []

    sessions: list[list[dict]] = []
    current_session: list[dict] = []

    for entry in entries:
        history_len = entry.get("history_length", 0)
        ts_str = entry.get("timestamp", "")

        # Start a new session if history_length is 0
        if history_len == 0 and current_session:
            sessions.append(current_session)
            current_session = []

        # Also check time gap
        if current_session and ts_str:
            try:
                prev_ts = datetime.fromisoformat(
                    current_session[-1].get("timestamp", "")
                )
                curr_ts = datetime.fromisoformat(ts_str)
                gap = (curr_ts - prev_ts).total_seconds()
                if gap > SESSION_GAP_SECONDS:
                    sessions.append(current_session)
                    current_session = []
            except (ValueError, TypeError):
                pass

        current_session.append(entry)

    if current_session:
        sessions.append(current_session)

    return sessions


def session_to_conversation(session: list[dict]) -> list[dict]:
    """Convert a session of log entries to a list of chat messages.

    Args:
        session: List of log entries from the same conversation.

    Returns:
        List of ``{"role": ..., "content": ...}`` dicts.
    """
    conversation = []
    for entry in session:
        conversation.append({"role": "user", "content": entry["user_message"]})
        conversation.append(
            {"role": "assistant", "content": entry["assistant_response"]}
        )
    return conversation


def conversation_to_alpaca(
    conversation: list[dict],
    system_prompt: str,
    *,
    metadata: dict | None = None,
) -> dict:
    """Convert a conversation to Alpaca-style training format.

    Uses the same format as ``generate_conversations.py``:
    - ``instruction``: system prompt
    - ``input``: last user message
    - ``output``: last assistant response
    - ``history``: all previous turns

    Args:
        conversation: List of chat messages.
        system_prompt: The OH system prompt.
        metadata: Optional metadata to attach.

    Returns:
        Alpaca-style training dict.
    """
    history = conversation[:-2] if len(conversation) > 2 else []
    last_user = conversation[-2]["content"]
    last_assistant = conversation[-1]["content"]

    result = {
        "instruction": system_prompt,
        "input": last_user,
        "output": last_assistant,
        "history": history,
    }

    if metadata:
        result["metadata"] = metadata

    return result


# ── Main pipeline ────────────────────────────────────────────────────


async def run(
    input_dir: Path,
    output_path: Path,
    *,
    score: bool = False,
    min_turns: int = 1,
) -> int:
    """Convert synced logs to training-ready JSONL.

    Args:
        input_dir: Path to synced companion logs directory.
        output_path: Output JSONL path.
        score: If True, use Groq to score and filter conversations.
        min_turns: Minimum number of turns to include.

    Returns:
        Number of conversations written.
    """
    # Load system prompt
    if V1_PROMPT_PATH.exists():
        system_prompt = V1_PROMPT_PATH.read_text(encoding="utf-8").strip()
        logger.info("Loaded system prompt (%d chars)", len(system_prompt))
    else:
        system_prompt = ""
        logger.warning("System prompt not found at %s — using empty", V1_PROMPT_PATH)

    # Load .env for Groq key if scoring
    if score:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())

    # Step 1: Load all logs
    raw_entries = load_all_logs(input_dir)
    if not raw_entries:
        logger.info("No log entries found — nothing to prepare.")
        return 0

    # Step 2: Filter to real interactions only
    real_entries = [e for e in raw_entries if is_real_interaction(e)]
    logger.info(
        "Filtered to %d real interactions (removed %d system calls)",
        len(real_entries),
        len(raw_entries) - len(real_entries),
    )

    if not real_entries:
        logger.info("No real interactions after filtering.")
        return 0

    # Step 3: Group into sessions
    sessions = group_into_sessions(real_entries)
    logger.info("Grouped into %d conversation sessions", len(sessions))

    # Step 4: Convert to conversations and filter by turn count
    conversations = []
    for session in sessions:
        conv = session_to_conversation(session)
        num_turns = len(conv) // 2
        if num_turns < min_turns:
            continue
        conversations.append(
            {
                "messages": conv,
                "num_turns": num_turns,
                "timestamps": [e.get("timestamp", "") for e in session],
                "model": session[0].get("model", "unknown"),
            }
        )

    logger.info(
        "%d conversations with >= %d turn(s)",
        len(conversations),
        min_turns,
    )

    # Step 5: Optional quality scoring
    groq_client = None
    if score:
        from groq import AsyncGroq

        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            logger.error("GROQ_API_KEY required for --score mode")
            sys.exit(1)
        groq_client = AsyncGroq(api_key=api_key, timeout=60.0)

    # Step 6: Convert to Alpaca format and write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    rejected = 0

    with open(output_path, "w") as outfile:
        for conv_info in conversations:
            conv = conv_info["messages"]

            # Score if requested
            if score and groq_client:
                s, reason = await score_conversation(conv, groq_client)
                if s < JUDGE_THRESHOLD:
                    logger.info(
                        "  Rejected (score %d/10): %s",
                        s,
                        reason[:60],
                    )
                    rejected += 1
                    continue
                metadata = {
                    "source": "real",
                    "score": s,
                    "reason": reason,
                    "num_turns": conv_info["num_turns"],
                    "model": conv_info["model"],
                    "timestamps": conv_info["timestamps"],
                }
            else:
                metadata = {
                    "source": "real",
                    "num_turns": conv_info["num_turns"],
                    "model": conv_info["model"],
                    "timestamps": conv_info["timestamps"],
                }

            alpaca = conversation_to_alpaca(conv, system_prompt, metadata=metadata)
            outfile.write(json.dumps(alpaca, ensure_ascii=False) + "\n")
            kept += 1

    logger.info(
        "Done: %d conversations kept, %d rejected → %s",
        kept,
        rejected,
        output_path,
    )
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert synced companion logs to training-ready JSONL",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory with synced JSONL log files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Score conversations with Groq judge (slower, more selective)",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=1,
        help="Minimum number of turns per conversation (default: 1)",
    )
    args = parser.parse_args()

    count = asyncio.run(
        run(
            args.input_dir,
            args.output,
            score=args.score,
            min_turns=args.min_turns,
        )
    )
    if count == 0:
        logger.info("No conversations produced.")
    else:
        logger.info("%d training conversations ready", count)


if __name__ == "__main__":
    main()
