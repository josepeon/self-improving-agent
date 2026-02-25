"""Tests for FineTuner.

MLX is optional and not available in CI, so we mock all mlx_lm imports
and only test the data formatting, config, and orchestration logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from self_improving_agent.training.finetune import FineTuner, LoRAConfig

# ── LoRAConfig ────────────────────────────────────────────────────────


class TestLoRAConfig:
    def test_defaults(self) -> None:
        cfg = LoRAConfig()
        assert cfg.rank == 16
        assert cfg.alpha == 32
        assert cfg.num_epochs == 3
        assert cfg.learning_rate == 1e-4
        assert "self_attn.q_proj" in cfg.target_modules

    def test_custom(self) -> None:
        cfg = LoRAConfig(rank=8, alpha=16, num_epochs=5)
        assert cfg.rank == 8
        assert cfg.num_epochs == 5


# ── Formatting ────────────────────────────────────────────────────────


class TestFormatForMLX:
    def test_basic_formatting(self) -> None:
        dataset = [
            {
                "instruction": "Be concise.",
                "input": "What is 2+2?",
                "output": "4",
                "history": [],
            }
        ]
        result = FineTuner._format_for_mlx(dataset)
        assert len(result) == 1
        msgs = result[0]["messages"]
        assert msgs[0] == {"role": "system", "content": "Be concise."}
        assert msgs[1] == {"role": "user", "content": "What is 2+2?"}
        assert msgs[2] == {"role": "assistant", "content": "4"}

    def test_with_history(self) -> None:
        dataset = [
            {
                "instruction": "System prompt.",
                "input": "Follow-up",
                "output": "Response",
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ],
            }
        ]
        result = FineTuner._format_for_mlx(dataset)
        msgs = result[0]["messages"]
        assert len(msgs) == 5  # system + 2 history + user + assistant
        assert msgs[1]["content"] == "Hello"
        assert msgs[2]["content"] == "Hi there"

    def test_no_instruction(self) -> None:
        dataset = [
            {
                "instruction": "",
                "input": "Hello",
                "output": "Hi",
                "history": [],
            }
        ]
        result = FineTuner._format_for_mlx(dataset)
        msgs = result[0]["messages"]
        # No system message when instruction is empty
        assert msgs[0]["role"] == "user"
        assert len(msgs) == 2

    def test_empty_dataset(self) -> None:
        assert FineTuner._format_for_mlx([]) == []


# ── Training orchestration ────────────────────────────────────────────


def _sample_dataset() -> list[dict[str, Any]]:
    return [
        {
            "instruction": "Be helpful.",
            "input": f"Question {i}",
            "output": f"Answer {i}",
            "history": [],
        }
        for i in range(5)
    ]


class TestFineTunerTrain:
    async def test_creates_version_directory(self, tmp_path: Path) -> None:
        ft = FineTuner(output_dir=tmp_path)
        with patch.object(ft, "_run_training", new_callable=AsyncMock):
            await ft.train(_sample_dataset(), "v1.0")
        version_dir = tmp_path / "v1.0"
        assert version_dir.exists()
        assert (version_dir / "train.jsonl").exists()
        assert (version_dir / "lora_config.json").exists()

    async def test_writes_correct_jsonl(self, tmp_path: Path) -> None:
        ft = FineTuner(output_dir=tmp_path)
        with patch.object(ft, "_run_training", new_callable=AsyncMock):
            await ft.train(_sample_dataset(), "v1.0")
        lines = (tmp_path / "v1.0" / "train.jsonl").read_text().strip().splitlines()
        assert len(lines) == 5
        first = json.loads(lines[0])
        assert "messages" in first

    async def test_saves_config(self, tmp_path: Path) -> None:
        cfg = LoRAConfig(rank=8, alpha=16)
        ft = FineTuner(output_dir=tmp_path, lora_config=cfg)
        with patch.object(ft, "_run_training", new_callable=AsyncMock):
            await ft.train(_sample_dataset(), "v1.0")
        config = json.loads((tmp_path / "v1.0" / "lora_config.json").read_text())
        assert config["rank"] == 8
        assert config["alpha"] == 16
        assert config["dataset_size"] == 5

    async def test_returns_adapter_path(self, tmp_path: Path) -> None:
        ft = FineTuner(output_dir=tmp_path)
        with patch.object(ft, "_run_training", new_callable=AsyncMock):
            result = await ft.train(_sample_dataset(), "v1.0")
        assert result == str(tmp_path / "v1.0" / "adapters")

    async def test_empty_dataset_raises(self, tmp_path: Path) -> None:
        ft = FineTuner(output_dir=tmp_path)
        with pytest.raises(ValueError, match="empty"):
            await ft.train([], "v1.0")

    async def test_epoch_override(self, tmp_path: Path) -> None:
        ft = FineTuner(output_dir=tmp_path)
        with patch.object(ft, "_run_training", new_callable=AsyncMock) as mock_train:
            await ft.train(_sample_dataset(), "v1.0", epochs=10)
            mock_train.assert_awaited_once()
            call_kwargs = mock_train.call_args.kwargs
            assert call_kwargs["num_epochs"] == 10

    async def test_calls_run_training(self, tmp_path: Path) -> None:
        ft = FineTuner(output_dir=tmp_path)
        with patch.object(ft, "_run_training", new_callable=AsyncMock) as mock_train:
            await ft.train(_sample_dataset(), "v1.0")
            mock_train.assert_awaited_once()
            call_kwargs = mock_train.call_args.kwargs
            assert call_kwargs["num_epochs"] == 3  # default


# ── Generate with adapter ────────────────────────────────────────────


class TestGenerateWithAdapter:
    async def test_calls_mlx_generate(self) -> None:
        with patch(
            "self_improving_agent.training.finetune.FineTuner.generate_with_adapter",
            new_callable=AsyncMock,
            return_value="Generated text",
        ) as mock_gen:
            result = await mock_gen("/path/to/adapter", "Hello")
            assert result == "Generated text"
