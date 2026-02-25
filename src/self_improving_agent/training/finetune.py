"""LoRA fine-tuning on Apple Silicon via MLX.

Trains adapter weights for Llama 3.1 8B using LoRA, reading
Alpaca-style JSONL datasets produced by :class:`DatasetBuilder`.

All ``mlx`` imports are deferred so the module can be imported
on any platform (Linux CI, etc.) without requiring MLX.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """LoRA hyper-parameters optimised for M4 Max 48 GB."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 3
    warmup_steps: int = 50
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )


class FineTuner:
    """Fine-tune Llama 3.1 8B using LoRA on MLX.

    Parameters
    ----------
    base_model:
        HuggingFace model ID for the MLX-quantised base.
    output_dir:
        Root directory for training artefacts and adapters.
    lora_config:
        LoRA hyper-parameters. Defaults are tuned for M4 Max 48 GB.
    """

    def __init__(
        self,
        base_model: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        output_dir: str | Path = "models/finetuned",
        lora_config: LoRAConfig | None = None,
    ) -> None:
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.config = lora_config or LoRAConfig()

    # ── Training ──────────────────────────────────────────────────────

    async def train(
        self,
        dataset: list[dict[str, Any]],
        version: str,
        *,
        epochs: int | None = None,
    ) -> str:
        """Train a new LoRA adapter version.

        Parameters
        ----------
        dataset:
            Alpaca-style dicts (instruction, input, output, history).
        version:
            Version tag, e.g. ``"v1.0"``.
        epochs:
            Override the default epoch count from :class:`LoRAConfig`.

        Returns
        -------
        str
            Path to the saved adapter weights directory.
        """
        num_epochs = epochs if epochs is not None else self.config.num_epochs

        # Prepare output directory
        version_dir = self.output_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Write training JSONL
        train_path = version_dir / "train.jsonl"
        formatted = self._format_for_mlx(dataset)
        with open(train_path, "w") as f:
            for item in formatted:
                f.write(json.dumps(item, separators=(",", ":")) + "\n")

        if len(formatted) == 0:
            msg = "Dataset is empty - nothing to train on"
            raise ValueError(msg)

        logger.info(
            "Starting LoRA training: %d examples, %d epochs, version %s",
            len(formatted),
            num_epochs,
            version,
        )

        # Save config alongside training artefacts
        config_path = version_dir / "lora_config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "base_model": self.base_model,
                    "rank": self.config.rank,
                    "alpha": self.config.alpha,
                    "dropout": self.config.dropout,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "num_epochs": num_epochs,
                    "target_modules": self.config.target_modules,
                    "dataset_size": len(formatted),
                },
                f,
                indent=2,
            )

        # Run MLX LoRA training (blocking - wrap in executor)
        adapter_path = version_dir / "adapters"
        await self._run_training(
            train_path=train_path,
            adapter_path=adapter_path,
            num_epochs=num_epochs,
        )

        logger.info("Training complete. Adapters saved to %s", adapter_path)
        return str(adapter_path)

    async def _run_training(
        self,
        train_path: Path,
        adapter_path: Path,
        num_epochs: int,
    ) -> None:
        """Execute MLX LoRA training.

        Imports ``mlx_lm`` lazily so the module works on non-macOS
        platforms (imports will fail gracefully when called).
        """
        import asyncio

        def _train() -> None:
            from mlx_lm.tuner import train

            train(
                model=self.base_model,
                data=str(train_path),
                adapter_path=str(adapter_path),
                lora_rank=self.config.rank,
                lora_alpha=self.config.alpha,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                num_epochs=num_epochs,
            )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _train)

    # ── Inference with adapters ───────────────────────────────────────

    async def generate_with_adapter(
        self,
        adapter_path: str | Path,
        prompt: str,
        *,
        max_tokens: int = 512,
    ) -> str:
        """Generate text using base model + LoRA adapter.

        Useful for evaluating a freshly-trained adapter before
        promoting it.
        """
        import asyncio

        def _generate() -> str:
            from mlx_lm import generate, load

            model, tokenizer = load(
                self.base_model,
                adapter_path=str(adapter_path),
            )
            return generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _generate)

    # ── Formatting ────────────────────────────────────────────────────

    @staticmethod
    def _format_for_mlx(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Alpaca-style dicts to MLX chat format.

        MLX expects ``{"messages": [{"role": ..., "content": ...}, ...]}``
        """
        formatted: list[dict[str, Any]] = []
        for item in dataset:
            messages: list[dict[str, str]] = []

            # System prompt
            if item.get("instruction"):
                messages.append({"role": "system", "content": item["instruction"]})

            # Conversation history
            if item.get("history"):
                messages.extend(item["history"])

            # Current turn
            messages.append({"role": "user", "content": item["input"]})
            messages.append({"role": "assistant", "content": item["output"]})

            formatted.append({"messages": messages})
        return formatted
