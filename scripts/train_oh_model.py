#!/usr/bin/env python3
"""Fine-tune, evaluate, and register the OH-native model.

Reads the synthetic training data, runs LoRA fine-tuning, evaluates
the result, and registers it in the model registry.

Usage::

    python scripts/train_oh_model.py [--dataset data/training/oh_sft_v1.jsonl]
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

DEFAULT_DATASET = PROJECT_ROOT / "data" / "training" / "oh_sft_v1.jsonl"
DEFAULT_VERSION = "v1.0-oh"
BASE_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> list[dict]:
    """Load Alpaca-style JSONL dataset."""
    if not path.exists():
        logger.error("Dataset not found: %s", path)
        logger.error("Run generate_training_data.py first")
        sys.exit(1)

    dataset = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))

    logger.info("Loaded %d training examples from %s", len(dataset), path.name)
    return dataset


async def train_model(dataset: list[dict], version: str) -> str:
    """Run LoRA fine-tuning and return adapter path."""
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

    logger.info(
        "Starting LoRA training: %d examples, version %s", len(dataset), version
    )
    adapter_path = await finetuner.train(dataset, version)
    logger.info("Adapters saved to: %s", adapter_path)
    return adapter_path


async def evaluate_model(adapter_path: str) -> dict:
    """Run a quick smoke-test with the trained adapter."""
    from self_improving_agent.training import FineTuner

    finetuner = FineTuner(base_model=BASE_MODEL)

    # Load the OH system prompt used during training
    prompts_path = PROJECT_ROOT.parent / "oh-companion-backend" / "prompts" / "v1.txt"
    system_prompt = ""
    if prompts_path.exists():
        system_prompt = prompts_path.read_text().strip()
        logger.info(
            "Loaded system prompt from %s (%d chars)",
            prompts_path.name,
            len(system_prompt),
        )
    else:
        logger.warning(
            "System prompt not found at %s — evaluating without it", prompts_path
        )

    test_questions = [
        "What is OH?",
        "What does O-OH do?",
        "Tell me about the Commerce Runtime.",
        "How does spatial commerce work?",
        "Who are you?",
    ]

    results = {}
    logger.info("Running smoke-test evaluation on %d questions…", len(test_questions))

    for q in test_questions:
        answer = await finetuner.generate_with_adapter(
            adapter_path, q, system_prompt=system_prompt, max_tokens=200
        )
        results[q] = answer
        logger.info("  Q: %s", q)
        logger.info("  A: %s", answer[:200])
        logger.info("")

    return results


async def register_model(version: str, adapter_path: str, dataset_size: int) -> None:
    """Register the trained model as production in the registry."""
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
    logger.info("Registered %s as PRODUCTION in model registry", version)

    # Verify
    prod = registry.get_production()
    if prod and prod.version == version:
        logger.info("Confirmed: %s is the current production model", version)
    else:
        logger.warning("Registration verification failed!")


async def run(dataset_path: Path, version: str) -> None:
    """Full pipeline: train → evaluate → register."""
    # 1. Load dataset
    dataset = load_dataset(dataset_path)
    if len(dataset) < 10:
        logger.warning(
            "Only %d examples — fine-tuning may be suboptimal. "
            "Consider generating more training data.",
            len(dataset),
        )

    # 2. Train
    adapter_path = await train_model(dataset, version)

    # 3. Evaluate
    eval_results = await evaluate_model(adapter_path)

    # Save evaluation results
    eval_path = Path(adapter_path).parent / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    logger.info("Evaluation results saved to %s", eval_path)

    # 4. Register
    await register_model(version, adapter_path, len(dataset))

    logger.info("=" * 60)
    logger.info("OH-native model training complete!")
    logger.info("  Version: %s", version)
    logger.info("  Adapter: %s", adapter_path)
    logger.info("  Dataset: %d examples", len(dataset))
    logger.info("  Status:  PRODUCTION")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the OH-native model")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to training JSONL (default: data/training/oh_sft_v1.jsonl)",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help="Model version tag (default: v1.0-oh)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: 42 or $SIA_SEED).",
    )
    args = parser.parse_args()

    # Pin every RNG SIA touches before any data shuffling or weight init.
    from self_improving_agent.reproducibility import get_seed_from_env, set_seed

    seed = args.seed if args.seed is not None else get_seed_from_env()
    set_seed(seed)

    asyncio.run(run(args.dataset, args.version))


if __name__ == "__main__":
    main()
