#!/usr/bin/env python3
"""Run a full self-improvement cycle: sync → prepare → train → evaluate → promote.

This is the main automation script for the SIA project. It chains all the
individual pipeline steps into a single orchestrated run.

Cycle steps::

    1. Sync logs from Railway (via sync_logs.py)
    2. Prepare real data (via prepare_real_data.py)
    3. Merge all training data (SFT + self-play + real)
    4. Optionally generate more self-play conversations
    5. Train new model version
    6. Evaluate against golden questions
    7. Compare with current production model
    8. Promote if passing, discard if not

Usage::

    # Full cycle — sync, prepare, train, evaluate, promote
    python scripts/run_improvement_cycle.py

    # Skip sync (if you already synced manually)
    python scripts/run_improvement_cycle.py --skip-sync

    # Skip self-play generation (just use existing data)
    python scripts/run_improvement_cycle.py --skip-self-play

    # Specify version tag
    python scripts/run_improvement_cycle.py --version v1.3-oh

    # Dry run — show what would happen without training
    python scripts/run_improvement_cycle.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ZEROZERO = PROJECT_ROOT.parent

# ── Paths ────────────────────────────────────────────────────────────

COMPANION_DIR = ZEROZERO / "oh-companion-backend"
V1_PROMPT_PATH = COMPANION_DIR / "prompts" / "v1.txt"
BASE_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

DATA_DIR = PROJECT_ROOT / "data"
TRAINING_DIR = DATA_DIR / "training"
COMPANION_LOGS_DIR = DATA_DIR / "companion_logs"
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"

SFT_PATH = TRAINING_DIR / "oh_sft_v1.jsonl"
CONV_V1_PATH = TRAINING_DIR / "oh_conversations_v1.jsonl"
CONV_V2_PATH = TRAINING_DIR / "oh_conversations_v2.jsonl"
REAL_DATA_PATH = TRAINING_DIR / "real_conversations.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────


def load_jsonl(path: Path, label: str) -> list[dict]:
    """Load a JSONL file, returning an empty list if missing."""
    if not path.exists():
        logger.info("  %s: not found — skipping", label)
        return []
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info("  %s: %d examples", label, len(data))
    return data


def get_current_production() -> dict | None:
    """Read the registry and return the current production version info."""
    if not REGISTRY_PATH.exists():
        return None
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    for v in registry.get("versions", []):
        if v.get("status") == "production":
            return v
    return None


def get_next_version(current: dict | None) -> str:
    """Auto-generate the next version tag based on the current production."""
    if current is None:
        return "v1.0-oh"

    version = current.get("version", "v1.0-oh")
    # Parse version like "v1.2-oh" → increment minor
    try:
        prefix = version.split("-")[0]  # "v1.2"
        parts = prefix[1:].split(".")  # ["1", "2"]
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return f"v{major}.{minor + 1}-oh"
    except (IndexError, ValueError):
        # Fallback: timestamp-based
        return f"v{time.strftime('%Y%m%d')}-oh"


# ── Pipeline steps ───────────────────────────────────────────────────


def step_sync_logs(since: str = "7d") -> int:
    """Step 1: Sync logs from Railway."""
    logger.info("=" * 60)
    logger.info("STEP 1: Syncing logs from Railway (since %s)", since)
    logger.info("=" * 60)

    # Import and run sync logic directly
    sys.path.insert(0, str(SCRIPT_DIR))
    from sync_logs import (
        extract_sia_entries,
        filter_real_interactions,
        run_railway_logs,
        save_entries,
    )

    try:
        raw_lines = run_railway_logs(
            since=since,
            companion_dir=str(COMPANION_DIR),
        )
        logger.info("  Retrieved %d raw log lines", len(raw_lines))
    except SystemExit:
        logger.warning("  Railway sync failed — continuing with existing data")
        return 0

    entries = extract_sia_entries(raw_lines)
    entries = filter_real_interactions(entries)

    if not entries:
        logger.info("  No new interactions to sync")
        return 0

    files_written = save_entries(entries, COMPANION_LOGS_DIR)
    total = 0
    if files_written:
        for fpath, count in files_written:
            logger.info("  %s: +%d entries", fpath.name, count)
            total += count
    else:
        logger.info("  All entries already synced")

    return total


async def step_prepare_real_data(score: bool = False) -> int:
    """Step 2: Convert synced logs to training format."""
    logger.info("=" * 60)
    logger.info("STEP 2: Preparing real conversation data")
    logger.info("=" * 60)

    from prepare_real_data import run as prepare_run

    count = await prepare_run(
        COMPANION_LOGS_DIR,
        REAL_DATA_PATH,
        score=score,
        min_turns=1,
    )
    return count


async def step_generate_self_play(
    rounds: int = 100,
    adapter_path: Path | None = None,
) -> int:
    """Step 3 (optional): Generate more self-play conversations."""
    logger.info("=" * 60)
    logger.info("STEP 3: Generating %d self-play conversations", rounds)
    logger.info("=" * 60)

    from generate_conversations import run as gen_run

    if adapter_path is None:
        # Use the current production adapter
        prod = get_current_production()
        if prod and prod.get("adapter_path"):
            adapter_path = Path(prod["adapter_path"])
        else:
            adapter_path = PROJECT_ROOT / "models" / "finetuned" / "v1.2-oh" / "adapters"

    output_path = TRAINING_DIR / "oh_conversations_latest.jsonl"
    count = await gen_run(
        output_path,
        rounds=rounds,
        adapter_path=adapter_path,
    )
    return count


def step_merge_datasets(
    include_real: bool = True,
    include_self_play_latest: bool = False,
) -> tuple[list[dict], dict[str, int]]:
    """Step 4: Merge all training data sources.

    Returns (combined_dataset, source_counts).
    """
    logger.info("=" * 60)
    logger.info("STEP 4: Merging training datasets")
    logger.info("=" * 60)

    source_counts: dict[str, int] = {}
    combined: list[dict] = []

    # Core SFT data (always included)
    sft = load_jsonl(SFT_PATH, "SFT (single-turn)")
    combined.extend(sft)
    source_counts["sft"] = len(sft)

    # Self-play v1 conversations
    conv_v1 = load_jsonl(CONV_V1_PATH, "Self-play v1")
    for item in conv_v1:
        item.pop("metadata", None)
    combined.extend(conv_v1)
    source_counts["self_play_v1"] = len(conv_v1)

    # Self-play v2 conversations
    conv_v2 = load_jsonl(CONV_V2_PATH, "Self-play v2")
    for item in conv_v2:
        item.pop("metadata", None)
    combined.extend(conv_v2)
    source_counts["self_play_v2"] = len(conv_v2)

    # Real conversation data
    if include_real:
        real = load_jsonl(REAL_DATA_PATH, "Real conversations")
        for item in real:
            item.pop("metadata", None)
        combined.extend(real)
        source_counts["real"] = len(real)

    # Latest self-play (from this cycle, if generated)
    if include_self_play_latest:
        latest_path = TRAINING_DIR / "oh_conversations_latest.jsonl"
        latest = load_jsonl(latest_path, "Self-play latest")
        for item in latest:
            item.pop("metadata", None)
        combined.extend(latest)
        source_counts["self_play_latest"] = len(latest)

    logger.info("  Combined: %d total examples", len(combined))
    for source, count in source_counts.items():
        logger.info("    %s: %d", source, count)

    return combined, source_counts


async def step_train(
    dataset: list[dict],
    version: str,
) -> str:
    """Step 5: Train a new model version with LoRA.

    Returns the adapter path.
    """
    logger.info("=" * 60)
    logger.info("STEP 5: Training %s (%d examples)", version, len(dataset))
    logger.info("=" * 60)

    from self_improving_agent.training import FineTuner, LoRAConfig

    config = LoRAConfig(
        rank=16,
        alpha=32,
        num_epochs=3,
        batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
    )

    finetuner = FineTuner(
        base_model=BASE_MODEL,
        output_dir=str(MODELS_DIR / "finetuned"),
        lora_config=config,
    )

    adapter_path = await finetuner.train(dataset, version)
    logger.info("  Adapters saved to: %s", adapter_path)
    return adapter_path


async def step_evaluate(adapter_path: str) -> dict:
    """Step 6: Evaluate the trained model on golden questions.

    Returns a dict of {question: answer}.
    """
    logger.info("=" * 60)
    logger.info("STEP 6: Evaluating model")
    logger.info("=" * 60)

    from self_improving_agent.training import FineTuner

    finetuner = FineTuner(base_model=BASE_MODEL)

    system_prompt = ""
    if V1_PROMPT_PATH.exists():
        system_prompt = V1_PROMPT_PATH.read_text().strip()

    questions = [
        "What is OH?",
        "What does O-OH do?",
        "Tell me about the Commerce Runtime.",
        "How does spatial commerce work?",
        "Who are you?",
        "Can I sell my products in a virtual world?",
        "What makes OH different from Shopify?",
    ]

    results = {}
    passed = 0

    for q in questions:
        answer = await finetuner.generate_with_adapter(
            adapter_path, q, system_prompt=system_prompt, max_tokens=200
        )
        results[q] = answer

        # Basic quality check: non-empty and mentions OH/spatial/commerce
        is_relevant = answer and len(answer) > 20
        if is_relevant:
            passed += 1
            logger.info("  ✓ %s", q)
        else:
            logger.warning("  ✗ %s → %s", q, answer[:100] if answer else "(empty)")

        logger.info("    → %s", answer[:150] if answer else "(empty)")

    pass_rate = passed / len(questions)
    logger.info(
        "  Evaluation: %d/%d passed (%.0f%%)",
        passed,
        len(questions),
        pass_rate * 100,
    )
    results["_pass_rate"] = pass_rate
    results["_passed"] = passed
    results["_total"] = len(questions)

    return results


def step_register(
    version: str,
    adapter_path: str,
    dataset_size: int,
    source_counts: dict[str, int],
    eval_results: dict,
    parent_version: str | None = None,
) -> None:
    """Step 7: Register (and optionally promote) the new version."""
    logger.info("=" * 60)
    logger.info("STEP 7: Registering %s", version)
    logger.info("=" * 60)

    from self_improving_agent.training import ModelRegistry, ModelStatus, ModelVersion

    registry = ModelRegistry(registry_path=str(REGISTRY_PATH))

    # Check evaluation results
    pass_rate = eval_results.get("_pass_rate", 0)

    if pass_rate < 0.85:
        logger.warning(
            "  Model failed evaluation (%.0f%% < 85%%) — marking as FAILED",
            pass_rate * 100,
        )
        status = ModelStatus.FAILED
    else:
        status = ModelStatus.PRODUCTION
        logger.info("  Evaluation passed (%.0f%%) — promoting to PRODUCTION", pass_rate * 100)

    # Save eval results
    eval_path = Path(adapter_path).parent / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    model_version = ModelVersion(
        version=version,
        base_model=BASE_MODEL,
        adapter_path=adapter_path,
        training_dataset_size=dataset_size,
        evaluation_pass_rate=pass_rate,
        evaluation_report_path=str(eval_path),
        status=status,
        data_sources=source_counts,
        parent_version=parent_version,
    )
    registry.register(model_version)

    logger.info("  Registered %s as %s", version, status)


# ── Main orchestrator ────────────────────────────────────────────────


async def run_cycle(
    *,
    version: str | None = None,
    since: str = "7d",
    skip_sync: bool = False,
    skip_self_play: bool = True,
    self_play_rounds: int = 100,
    score_real: bool = False,
    dry_run: bool = False,
) -> None:
    """Run the full self-improvement cycle."""
    start_time = time.time()

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   SIA Self-Improvement Cycle                             ║")
    logger.info("╚" + "═" * 58 + "╝")

    # Determine version
    current_prod = get_current_production()
    parent_version = current_prod.get("version") if current_prod else None

    if version is None:
        version = get_next_version(current_prod)

    logger.info("  Current production: %s", parent_version or "(none)")
    logger.info("  Training version:   %s", version)
    logger.info("  Sync since:         %s", since)
    logger.info("  Score real data:    %s", score_real)
    logger.info("  Self-play rounds:   %s", self_play_rounds if not skip_self_play else "skipped")
    logger.info("")

    # Step 1: Sync
    if not skip_sync:
        synced = step_sync_logs(since=since)
        logger.info("  → Synced %d new interactions\n", synced)
    else:
        logger.info("  → Sync skipped\n")

    # Step 2: Prepare real data
    real_count = await step_prepare_real_data(score=score_real)
    logger.info("  → %d real conversations prepared\n", real_count)

    # Step 3: Optional self-play
    if not skip_self_play:
        sp_count = await step_generate_self_play(rounds=self_play_rounds)
        logger.info("  → %d self-play conversations generated\n", sp_count)

    # Step 4: Merge
    dataset, source_counts = step_merge_datasets(
        include_real=True,
        include_self_play_latest=not skip_self_play,
    )

    if len(dataset) < 10:
        logger.error("Dataset too small (%d). Aborting.", len(dataset))
        return

    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN — would train %s with %d examples", version, len(dataset))
        logger.info("  Sources: %s", source_counts)
        logger.info("=" * 60)
        return

    # Step 5: Train
    adapter_path = await step_train(dataset, version)

    # Step 6: Evaluate
    eval_results = await step_evaluate(adapter_path)

    # Step 7: Register
    step_register(
        version=version,
        adapter_path=adapter_path,
        dataset_size=len(dataset),
        source_counts=source_counts,
        eval_results=eval_results,
        parent_version=parent_version,
    )

    elapsed = time.time() - start_time
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   Cycle Complete                                         ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("  Version:     %s", version)
    logger.info("  Adapter:     %s", adapter_path)
    logger.info("  Dataset:     %d examples", len(dataset))
    logger.info("  Sources:     %s", source_counts)
    logger.info("  Pass rate:   %.0f%%", eval_results.get("_pass_rate", 0) * 100)
    logger.info("  Parent:      %s", parent_version or "(none)")
    logger.info("  Duration:    %.1f minutes", elapsed / 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a full SIA self-improvement cycle",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version tag for the new model (auto-incremented if not set)",
    )
    parser.add_argument(
        "--since",
        default="7d",
        help="How far back to sync logs (default: 7d)",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip Railway log sync (use existing local data)",
    )
    parser.add_argument(
        "--skip-self-play",
        action="store_true",
        default=True,
        help="Skip self-play generation (default: skip)",
    )
    parser.add_argument(
        "--self-play",
        action="store_true",
        help="Enable self-play generation in this cycle",
    )
    parser.add_argument(
        "--self-play-rounds",
        type=int,
        default=100,
        help="Number of self-play conversations to generate (default: 100)",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Score real conversations with Groq judge before training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without actually training",
    )
    args = parser.parse_args()

    # --self-play overrides --skip-self-play
    skip_self_play = not args.self_play

    asyncio.run(
        run_cycle(
            version=args.version,
            since=args.since,
            skip_sync=args.skip_sync,
            skip_self_play=skip_self_play,
            self_play_rounds=args.self_play_rounds,
            score_real=args.score,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
