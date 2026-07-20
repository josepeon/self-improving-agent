"""``sia train`` — invoke the self-improvement pipeline.

Thin wrapper over ``SelfImprovementPipeline.run()`` with sane defaults and
flag overrides. The pipeline itself handles dataset assembly, fine-tuning,
evaluation, and registry promotion.

For the lower-level training scripts (``scripts/train_oh_model_*.py``)
this is a coordination layer, not a replacement — those produce the
initial adapters; this command runs the *improvement loop* over them.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from self_improving_agent.pipeline.self_improve import (
    PipelineConfig,
    SelfImprovementPipeline,
)
from self_improving_agent.reproducibility import set_seed
from self_improving_agent.training.finetune import LoRAConfig

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--registry",
        default="models/registry.json",
        help="Registry path (default: models/registry.json)",
    )
    parser.add_argument(
        "--log-base-path",
        default="logs",
        help="Interaction log root (default: logs/)",
    )
    parser.add_argument(
        "--finetuned-dir",
        default="models/finetuned",
        help="Adapter output root (default: models/finetuned/)",
    )
    parser.add_argument(
        "--reports-dir",
        default="evaluation/reports",
        help="Eval report output (default: evaluation/reports/)",
    )
    parser.add_argument(
        "--min-dataset-size",
        type=int,
        default=100,
        help="Skip training if fewer than N high-quality examples are available.",
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.8,
        help="Drop training rows below this quality score (0..1).",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=0.95,
        help="New version must score at least N times the production pass rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (mirrors T3.2 behaviour).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the dataset and report what *would* train, then exit.",
    )


def run(args: argparse.Namespace) -> int:
    cfg = PipelineConfig(
        log_base_path=args.log_base_path,
        registry_path=args.registry,
        reports_dir=args.reports_dir,
        finetuned_dir=args.finetuned_dir,
        min_dataset_size=args.min_dataset_size,
        min_quality_score=args.min_quality_score,
        regression_threshold=args.regression_threshold,
        lora_config=LoRAConfig(),
    )

    # Pipeline expects a model loader callable. For the CLI we wire MLXProvider.
    async def model_loader(version):
        from self_improving_agent.providers.base import GenerationConfig
        from self_improving_agent.providers.mlx_provider import MLXProvider

        provider = MLXProvider(
            model_path=version.base_model,
            adapter_path=version.adapter_path,
        )

        async def generate_fn(
            messages, *, temperature: float = 0.0, max_tokens: int = 256
        ):
            cfg = GenerationConfig(temperature=temperature, max_tokens=max_tokens)
            r = await provider.generate(messages, cfg)
            return r.text

        return generate_fn

    pipeline = SelfImprovementPipeline(config=cfg, model_loader=model_loader)

    if args.dry_run:
        return _dry_run(pipeline, cfg)

    set_seed(args.seed)
    print("Starting improvement cycle (this may take hours)...")
    version = asyncio.run(pipeline.run_weekly_training())

    if version is None:
        print("No training run: not enough high-quality data yet.")
        return 0

    pass_rate = (
        f"{version.evaluation_pass_rate:.0%}"
        if version.evaluation_pass_rate is not None
        else "n/a"
    )
    print(
        f"\nTrained {version.version} "
        f"(status: {version.status}, pass rate: {pass_rate})"
    )
    return 0


def _dry_run(pipeline: SelfImprovementPipeline, cfg: PipelineConfig) -> int:
    """Inspect what the next cycle would do without actually training."""
    try:
        stats = asyncio.run(pipeline.dataset_stats())
    except Exception as exc:
        print(f"Could not read interaction log: {exc}")
        return 1

    print(f"Interaction log: {stats['total_interactions']} row(s) found")
    print(f"SFT-eligible:      {stats['sft_eligible']}")
    print(f"Quality threshold: {cfg.min_quality_score}")
    print(f"Min dataset size:  {cfg.min_dataset_size}")
    print(f"Output adapter:    {cfg.finetuned_dir}/<next-version>/")
    if stats["sft_eligible"] < cfg.min_dataset_size:
        print(
            "Below min_dataset_size — a real run would skip training "
            "until more data accumulates."
        )
    return 0
