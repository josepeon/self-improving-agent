"""T2.4: LoRA adapter merging."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from self_improving_agent.training.merging import (
    ADAPTER_CONFIG_FILENAME,
    ADAPTER_FILENAME,
    MergeSource,
    merge_adapters,
)


@pytest.fixture
def adapter_factory(tmp_path):
    """Returns a factory that writes a small fake adapter and returns its dir."""
    from safetensors.numpy import save_file

    def _make(name: str, tensors: dict[str, np.ndarray], config: dict | None = None):
        directory = tmp_path / name
        directory.mkdir()
        save_file(tensors, str(directory / ADAPTER_FILENAME))
        if config is not None:
            (directory / ADAPTER_CONFIG_FILENAME).write_text(json.dumps(config))
        return directory

    return _make


# ---------------------------------------------------------------------
# Basic merge
# ---------------------------------------------------------------------


class TestMergeAdapters:
    def test_two_adapter_average(self, adapter_factory, tmp_path):
        # Both adapters share a single key; merged value should equal the
        # weighted average.
        a = adapter_factory(
            "a",
            {"w": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
            config={"rank": 16, "alpha": 32},
        )
        b = adapter_factory(
            "b",
            {"w": np.array([3.0, 4.0, 5.0], dtype=np.float32)},
        )

        out = tmp_path / "merged"
        result = merge_adapters(
            [MergeSource(a, 0.5), MergeSource(b, 0.5)],
            out,
        )

        from safetensors.numpy import load_file

        merged = load_file(str(out / ADAPTER_FILENAME))
        np.testing.assert_allclose(merged["w"], [2.0, 3.0, 4.0])
        assert result.keys_merged == 1
        assert result.keys_only_in_one == []

    def test_three_adapter_weighted(self, adapter_factory, tmp_path):
        a = adapter_factory("a", {"w": np.array([1.0, 0.0], dtype=np.float32)})
        b = adapter_factory("b", {"w": np.array([0.0, 1.0], dtype=np.float32)})
        c = adapter_factory("c", {"w": np.array([1.0, 1.0], dtype=np.float32)})

        out = tmp_path / "out"
        merge_adapters(
            [MergeSource(a, 0.5), MergeSource(b, 0.25), MergeSource(c, 0.25)],
            out,
        )

        from safetensors.numpy import load_file

        merged = load_file(str(out / ADAPTER_FILENAME))
        # 0.5*[1,0] + 0.25*[0,1] + 0.25*[1,1] = [0.75, 0.5]
        np.testing.assert_allclose(merged["w"], [0.75, 0.5])

    def test_disjoint_keys_contribute_with_zero_for_missing(
        self, adapter_factory, tmp_path
    ):
        # Adapter a has key 'w1', adapter b has key 'w2'. Both should
        # appear in the merge, scaled only by their own contributor's weight.
        a = adapter_factory("a", {"w1": np.array([2.0], dtype=np.float32)})
        b = adapter_factory("b", {"w2": np.array([4.0], dtype=np.float32)})

        out = tmp_path / "out"
        result = merge_adapters(
            [MergeSource(a, 0.5), MergeSource(b, 0.5)],
            out,
        )

        from safetensors.numpy import load_file

        merged = load_file(str(out / ADAPTER_FILENAME))
        np.testing.assert_allclose(merged["w1"], [1.0])
        np.testing.assert_allclose(merged["w2"], [2.0])
        assert sorted(result.keys_only_in_one) == ["w1", "w2"]

    def test_normalize_weights(self, adapter_factory, tmp_path):
        # Weights 2.0 and 2.0 should normalise to 0.5 each
        a = adapter_factory("a", {"w": np.array([1.0], dtype=np.float32)})
        b = adapter_factory("b", {"w": np.array([3.0], dtype=np.float32)})

        out = tmp_path / "out"
        result = merge_adapters(
            [MergeSource(a, 2.0), MergeSource(b, 2.0)],
            out,
            normalize_weights=True,
        )

        from safetensors.numpy import load_file

        merged = load_file(str(out / ADAPTER_FILENAME))
        np.testing.assert_allclose(merged["w"], [2.0])
        # Source weights post-normalisation are reflected in the result
        assert result.sources[0].weight == pytest.approx(0.5)

    def test_zero_normalize_raises(self, adapter_factory, tmp_path):
        a = adapter_factory("a", {"w": np.array([1.0], dtype=np.float32)})
        b = adapter_factory("b", {"w": np.array([1.0], dtype=np.float32)})

        with pytest.raises(ValueError, match="sum to zero"):
            merge_adapters(
                [MergeSource(a, 0.0), MergeSource(b, 0.0)],
                tmp_path / "out",
                normalize_weights=True,
            )

    def test_empty_sources_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="At least one"):
            merge_adapters([], tmp_path / "out")

    def test_missing_safetensors_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            merge_adapters(
                [MergeSource(empty_dir, 1.0)],
                tmp_path / "out",
            )


# ---------------------------------------------------------------------
# Config carryover
# ---------------------------------------------------------------------


class TestMergeConfig:
    def test_carries_first_source_config(self, adapter_factory, tmp_path):
        a = adapter_factory(
            "a",
            {"w": np.array([1.0], dtype=np.float32)},
            config={"rank": 16, "alpha": 32, "base_model": "test/llama"},
        )
        b = adapter_factory("b", {"w": np.array([3.0], dtype=np.float32)})

        out = tmp_path / "out"
        merge_adapters([MergeSource(a, 0.5), MergeSource(b, 0.5)], out)

        with open(out / ADAPTER_CONFIG_FILENAME) as f:
            config = json.load(f)
        assert config["rank"] == 16
        assert config["base_model"] == "test/llama"
        assert config["merge_method"] == "weighted_sum"
        assert len(config["merged_from"]) == 2
        assert config["merged_from"][0]["weight"] == 0.5

    def test_no_config_anywhere_still_writes_merge_annotation(
        self, adapter_factory, tmp_path
    ):
        a = adapter_factory("a", {"w": np.array([1.0], dtype=np.float32)})
        out = tmp_path / "out"
        merge_adapters([MergeSource(a, 1.0)], out)

        with open(out / ADAPTER_CONFIG_FILENAME) as f:
            config = json.load(f)
        # No fields from a source config — but the merge annotation IS there
        assert config["merge_method"] == "weighted_sum"


# ---------------------------------------------------------------------
# Output preservation
# ---------------------------------------------------------------------


class TestOutputShape:
    def test_dtype_preserved(self, adapter_factory, tmp_path):
        a = adapter_factory(
            "a",
            {"w": np.array([1.0, 2.0], dtype=np.float16)},
        )
        b = adapter_factory(
            "b",
            {"w": np.array([3.0, 4.0], dtype=np.float16)},
        )
        out = tmp_path / "out"
        merge_adapters([MergeSource(a, 0.5), MergeSource(b, 0.5)], out)

        from safetensors.numpy import load_file

        merged = load_file(str(out / ADAPTER_FILENAME))
        assert merged["w"].dtype == np.float16
