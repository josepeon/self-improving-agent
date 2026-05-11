"""``sia merge`` — weighted-sum merge of N LoRA adapters into a new one.

Usage:
    sia merge --output models/finetuned/v1.3-oh-mixed/adapters \\
        --adapter v1.0-oh:0.6 \\
        --adapter v1.2-oh:0.4

The adapter spec is ``<registry-version|path>:<weight>``. Bare paths skip
the registry lookup so unregistered experimental adapters can be merged.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from self_improving_agent.training.merging import (
    MergeSource,
    merge_adapters,
)
from self_improving_agent.training.registry import ModelRegistry

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory to write the merged adapter into.",
    )
    parser.add_argument(
        "--adapter",
        action="append",
        required=True,
        help=(
            "Source adapter as '<version|path>:<weight>'. "
            "Repeatable. Example: --adapter v1.0-oh:0.6 --adapter v1.2-oh:0.4"
        ),
    )
    parser.add_argument(
        "--registry",
        default="models/registry.json",
        help="Registry path used to resolve version tags (default: models/registry.json)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Rescale weights to sum to 1.0 before the merge.",
    )


def run(args: argparse.Namespace) -> int:
    registry = ModelRegistry(args.registry) if Path(args.registry).exists() else None

    sources: list[MergeSource] = []
    for spec in args.adapter:
        if ":" not in spec:
            print(f"Invalid --adapter '{spec}'; expected '<version|path>:<weight>'")
            return 2
        target, _, weight_str = spec.rpartition(":")
        try:
            weight = float(weight_str)
        except ValueError:
            print(f"Invalid weight in '{spec}': {weight_str!r}")
            return 2

        path = _resolve_target(target, registry)
        if path is None:
            print(f"Could not resolve adapter source: {target!r}")
            return 2
        sources.append(MergeSource(path=path, weight=weight))

    try:
        result = merge_adapters(
            sources, args.output, normalize_weights=args.normalize
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Merge failed: {exc}")
        return 1

    print(f"Merged {len(sources)} adapter(s) -> {result.output_path}")
    print(f"  keys merged:        {result.keys_merged}")
    if result.keys_only_in_one:
        print(f"  keys only in one:   {len(result.keys_only_in_one)}")
        for k in result.keys_only_in_one[:5]:
            print(f"    - {k}")
        if len(result.keys_only_in_one) > 5:
            print(f"    ... and {len(result.keys_only_in_one) - 5} more")
    print("  config written to:  adapter_config.json")
    return 0


def _resolve_target(target: str, registry: ModelRegistry | None) -> Path | None:
    """Accept either a registry version tag or a filesystem path."""
    # Direct path first
    direct = Path(target)
    if direct.exists() and direct.is_dir():
        return direct

    if registry is None:
        return None

    entry = registry.get(target)
    if entry is None or not entry.adapter_path:
        return None

    candidate = Path(entry.adapter_path)
    if candidate.exists():
        return candidate
    return None
