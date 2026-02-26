#!/usr/bin/env python3
"""Train v1.2-oh from SFT + all conversation data (v1 + v2).

Merges the original SFT pairs with both rounds of self-play
conversations and trains a new LoRA adapter.

Usage::

    # After generating v2 conversations tonight:
    python scripts/train_oh_model_v12.py

    # Or with custom paths:
    python scripts/train_oh_model_v12.py --extra-conversations data/training/extra.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_SFT = PROJECT_ROOT / "data" / "training" / "oh_sft_v1.jsonl"
DEFAULT_CONV_V1 = PROJECT_ROOT / "data" / "training" / "oh_conversations_v1.jsonl"
DEFAULT_CONV_V2 = PROJECT_ROOT / "data" / "training" / "oh_conversations_v2.jsonl"
DEFAULT_VERSION = "v1.2-oh"
BASE_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


def load_jsonl(path: Path, label: str) -> list[dict]:
    """Load JSONL dataset, logging count."""
    if not path.exists():
        logger.warning("%s not found: %s — skipping", label, path)
        return []
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info("Loaded %d examples from %s (%s)", len(data), path.name, label)
    return data


async def train_model(dataset: list[dict], version: str) -> str:
    """Run LoRA fine-tuning."""
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
        output_dir=str(PROJECT_ROOT / "models" / "finetuned"),
        lora_config=config,
    )

    logger.info("Starting training: %d examples, version %s", len(dataset), version)
    adapter_path = await finetuner.train(dataset, version)
    logger.info("Adapters saved to: %s", adapter_path)
    return adapter_path


async def evaluate_model(adapter_path: str) -> dict:
    """Smoke-test the trained model."""
    from self_improving_agent.training import FineTuner

    finetuner = FineTuner(base_model=BASE_MODEL)

    prompts_path = PROJECT_ROOT.parent / "oh-companion-backend" / "prompts" / "v1.txt"
    system_prompt = ""
    if prompts_path.exists():
        system_prompt = prompts_path.read_text().strip()

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
    logger.info("Running evaluation on %d questions…", len(questions))
    for q in questions:
        answer = await finetuner.generate_with_adapter(
            adapter_path, q, system_prompt=system_prompt, max_tokens=200
        )
        results[q] = answer
        logger.info("  Q: %s", q)
        logger.info("  A: %s", answer[:200])
        logger.info("")
    return results


async def register_model(version: str, adapter_path: str, dataset_size: int) -> None:
    """Register as production, demoting the previous version."""
    from self_improving_agent.training import ModelRegistry, ModelStatus, ModelVersion

    registry = ModelRegistry(
        registry_path=str(PROJECT_ROOT / "models" / "registry.json")
    )

    model_version = ModelVersion(
        version=version,
        base_model=BASE_MODEL,
        adapter_path=adapter_path,
        training_dataset_size=dataset_size,
        status=ModelStatus.PRODUCTION,
    )
    registry.register(model_version)
    logger.info("Registered %s as PRODUCTION", version)


async def run(
    sft_path: Path,
    conv_paths: list[Path],
    version: str,
) -> None:
    """Full pipeline: merge data → train → evaluate → register."""
    # 1. Load and merge datasets
    sft_data = load_jsonl(sft_path, "single-turn SFT")

    all_conv_data = []
    for conv_path in conv_paths:
        conv_data = load_jsonl(conv_path, f"conversations ({conv_path.stem})")
        # Strip metadata from conversation data (not needed for training)
        for item in conv_data:
            item.pop("metadata", None)
        all_conv_data.extend(conv_data)

    combined = sft_data + all_conv_data
    if len(combined) < 10:
        logger.error(
            "Combined dataset too small (%d). Generate more data first.",
            len(combined),
        )
        sys.exit(1)

    logger.info(
        "Combined dataset: %d total (%d single-turn + %d multi-turn)",
        len(combined),
        len(sft_data),
        len(all_conv_data),
    )

    # 2. Train
    adapter_path = await train_model(combined, version)

    # 3. Evaluate
    eval_results = await evaluate_model(adapter_path)
    eval_path = Path(adapter_path).parent / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    logger.info("Evaluation results saved to %s", eval_path)

    # 4. Register
    await register_model(version, adapter_path, len(combined))

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("  Version:     %s", version)
    logger.info("  Adapter:     %s", adapter_path)
    logger.info("  Single-turn: %d examples", len(sft_data))
    logger.info("  Multi-turn:  %d conversations", len(all_conv_data))
    logger.info("  Total:       %d", len(combined))
    logger.info("  Status:      PRODUCTION")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train v1.2-oh from SFT + all conversation rounds"
    )
    parser.add_argument(
        "--sft",
        type=Path,
        default=DEFAULT_SFT,
        help="Single-turn SFT JSONL",
    )
    parser.add_argument(
        "--conversations-v1",
        type=Path,
        default=DEFAULT_CONV_V1,
        help="V1 conversation JSONL (200 rounds from v1.0-oh)",
    )
    parser.add_argument(
        "--conversations-v2",
        type=Path,
        default=DEFAULT_CONV_V2,
        help="V2 conversation JSONL (500 rounds from v1.1-oh)",
    )
    parser.add_argument(
        "--extra-conversations",
        type=Path,
        default=None,
        help="Additional conversation JSONL (e.g., real user data)",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help="Model version tag (default: v1.2-oh)",
    )
    args = parser.parse_args()

    conv_paths = [args.conversations_v1, args.conversations_v2]
    if args.extra_conversations:
        conv_paths.append(args.extra_conversations)

    asyncio.run(run(args.sft, conv_paths, args.version))


if __name__ == "__main__":
    main()
