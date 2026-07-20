"""
Reproducibility seeding

Pins every RNG the training scripts and self-play data generator touch:
Python's ``random``, NumPy, MLX (if installed). One call at the start of
a script makes the run repeatable.

What this can and can't guarantee:

  - Repeatable: dataset shuffles, batch ordering, MLX seed-controlled
    sampling, temperature-based sampling decisions, weight init for LoRA.
  - NOT repeatable without further work: MLX inference under
    gradient-checkpointing + half-precision (small drift each step),
    multi-threaded data loading order (mlx-lm uses single-threaded by
    default), and the order of LLM API responses from Groq (cloud-side).

The seed is also stamped into ``ModelVersion.metadata`` by the training
scripts so reports can be traced back to the exact RNG state.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42


@dataclass
class SeedReport:
    """Returned by ``set_seed`` so scripts can stamp the seed into metadata."""

    seed: int
    python: bool = False
    numpy: bool = False
    mlx: bool = False
    pythonhashseed: bool = False

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "python": self.python,
            "numpy": self.numpy,
            "mlx": self.mlx,
            "pythonhashseed": self.pythonhashseed,
        }


def set_seed(seed: int = DEFAULT_SEED) -> SeedReport:
    """Seed every RNG SIA touches. Returns a SeedReport.

    Args:
        seed: Integer seed to use across all RNGs. Defaults to 42.

    Returns:
        SeedReport indicating which libraries were successfully seeded.
        Optional dependencies that aren't installed (mlx, numpy) are
        reported as False without raising.
    """
    report = SeedReport(seed=seed)

    # PYTHONHASHSEED — important for any hash-order-dependent code. Note:
    # only takes effect on a fresh process; setting it mid-run has no
    # retroactive effect on already-initialised hashes. We still set it so
    # subprocesses inherit it.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    report.pythonhashseed = True

    # Standard library
    random.seed(seed)
    report.python = True

    # NumPy
    try:
        import numpy as np

        np.random.seed(seed)
        report.numpy = True
    except ImportError:
        pass

    # MLX (Apple Silicon)
    try:
        import mlx.core as mx

        mx.random.seed(seed)
        report.mlx = True
    except ImportError:
        pass
    except Exception as exc:
        # MLX exists but seeding failed somehow — log, don't crash.
        logger.warning("mlx.random.seed failed: %s", exc)

    logger.info("Reproducibility: seed=%d %s", seed, report.as_dict())
    return report


def get_seed_from_env(default: int = DEFAULT_SEED) -> int:
    """Helper for scripts: read ``SIA_SEED`` env or fall back to default."""
    raw = os.environ.get("SIA_SEED", "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("SIA_SEED=%r is not a valid integer; using %d", raw, default)
        return default
