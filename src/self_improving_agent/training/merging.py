"""
LoRA adapter merging

Each training run produces a separate adapter. There's no way to combine
"adapter trained on general OH knowledge" with "adapter trained on
Bianca-specific brand voice" without retraining from scratch.

This module implements weighted task-arithmetic merging on the safetensors
weight files. Math is intentionally simple:

    merged_weight[k] = Σ_i (w_i * adapter_i[k])

where w_i are the user-supplied weights (commonly summing to 1.0 but not
required). Adapters that don't share a key contribute 0 for that key in
the missing adapter — equivalent to padding with zeros.

We deliberately don't implement TIES-merging or DARE here. Those need
sign-conflict resolution and stochastic dropout passes that are easier
to get wrong than weighted-sum. Add later if a use case demands it.

The merger reads/writes ``.safetensors`` directly (we don't need MLX
loaded to do tensor arithmetic on the file contents).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ADAPTER_FILENAME = "adapters.safetensors"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"


@dataclass
class MergeSource:
    """One input to the merger."""

    path: Path  # directory containing adapters.safetensors
    weight: float


@dataclass
class MergeResult:
    """End-of-merge summary."""

    output_path: Path
    sources: list[MergeSource]
    keys_merged: int = 0
    keys_only_in_one: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Tensor IO (safetensors)
# ---------------------------------------------------------------------


def _load_tensors(path: Path) -> dict[str, Any]:
    """Load all tensors from a safetensors file as a dict of arrays.

    Lazy-imports safetensors and numpy. Tries the numpy load path first
    (no GPU framework needed); falls back to torch only if numpy can't
    handle the dtype (rare for LoRA weights).
    """
    try:
        from safetensors.numpy import load_file
    except ImportError as exc:
        raise ImportError(
            "Adapter merging requires the safetensors package. "
            "Install with: pip install safetensors numpy"
        ) from exc

    return dict(load_file(str(path)))


def _save_tensors(path: Path, tensors: dict[str, Any]) -> None:
    from safetensors.numpy import save_file

    save_file(tensors, str(path))


# ---------------------------------------------------------------------
# Core merge
# ---------------------------------------------------------------------


def merge_adapters(
    sources: list[MergeSource],
    output_dir: Path,
    *,
    normalize_weights: bool = False,
) -> MergeResult:
    """Weighted-sum-merge ``sources`` into ``output_dir``.

    Args:
        sources: list of MergeSource(path=..., weight=...).
        output_dir: written; created if missing. The merger writes both
            ``adapters.safetensors`` and a merged ``adapter_config.json``
            (rank/alpha taken from the first source).
        normalize_weights: if True, weights are rescaled to sum to 1.0
            before the merge. Useful for ensembling. Default False.

    Returns:
        MergeResult with key counts and the resolved config.
    """
    if not sources:
        raise ValueError("At least one source is required")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = [s.weight for s in sources]
    if normalize_weights:
        total = sum(weights)
        if total == 0:
            raise ValueError("Cannot normalize: weights sum to zero")
        weights = [w / total for w in weights]

    # Load every adapter
    loaded: list[tuple[MergeSource, float, dict[str, Any]]] = []
    for source, weight in zip(sources, weights, strict=True):
        adapter_file = source.path / ADAPTER_FILENAME
        if not adapter_file.exists():
            raise FileNotFoundError(f"No {ADAPTER_FILENAME} in {source.path}")
        loaded.append((source, weight, _load_tensors(adapter_file)))

    # Union of all keys
    all_keys: set[str] = set()
    for _, _, tensors in loaded:
        all_keys.update(tensors.keys())

    # Merge
    import numpy as np

    merged: dict[str, Any] = {}
    only_in_one: list[str] = []
    keys_per_count: dict[str, int] = {}
    for key in all_keys:
        contributors = [
            (weight, tensors[key]) for _, weight, tensors in loaded if key in tensors
        ]
        keys_per_count[key] = len(contributors)
        if len(contributors) == 1:
            only_in_one.append(key)

        # Reduce: start with weighted first, add others
        first_weight, first_tensor = contributors[0]
        result = first_weight * first_tensor.astype(np.float32)
        for w, t in contributors[1:]:
            result = result + w * t.astype(np.float32)
        # Cast back to the original dtype of the first contributor
        merged[key] = result.astype(first_tensor.dtype)

    _save_tensors(output_dir / ADAPTER_FILENAME, merged)

    # Carry over the first source's adapter_config.json, annotated with the merge
    config = _read_first_config(sources[0].path)
    config["merged_from"] = [
        {"path": str(s.path), "weight": w}
        for s, w in zip(sources, weights, strict=True)
    ]
    config["merge_method"] = "weighted_sum"
    with open(output_dir / ADAPTER_CONFIG_FILENAME, "w") as f:
        json.dump(config, f, indent=2)

    return MergeResult(
        output_path=output_dir,
        sources=[
            MergeSource(path=s.path, weight=w)
            for s, w in zip(sources, weights, strict=True)
        ],
        keys_merged=len(merged),
        keys_only_in_one=only_in_one,
        config=config,
    )


def _read_first_config(adapter_dir: Path) -> dict[str, Any]:
    """Best-effort read of the first source's adapter_config.json."""
    candidate = adapter_dir / ADAPTER_CONFIG_FILENAME
    if candidate.exists():
        try:
            with open(candidate) as f:
                loaded: dict[str, Any] = json.load(f)
                return loaded
        except json.JSONDecodeError:
            pass

    # Some training runs put the config in the *parent* directory
    parent_candidate = adapter_dir.parent / "lora_config.json"
    if parent_candidate.exists():
        try:
            with open(parent_candidate) as f:
                parent_loaded: dict[str, Any] = json.load(f)
                return parent_loaded
        except json.JSONDecodeError:
            pass

    return {}
