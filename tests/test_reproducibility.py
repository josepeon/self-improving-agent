"""T3.2: reproducibility seeding."""

from __future__ import annotations

import os
import random

import pytest

from self_improving_agent.reproducibility import (
    DEFAULT_SEED,
    SeedReport,
    get_seed_from_env,
    set_seed,
)


class TestSetSeed:
    def test_returns_report(self):
        report = set_seed(123)
        assert isinstance(report, SeedReport)
        assert report.seed == 123
        assert report.python is True
        assert report.pythonhashseed is True

    def test_seeding_python_random_is_deterministic(self):
        set_seed(7)
        seq1 = [random.random() for _ in range(5)]
        set_seed(7)
        seq2 = [random.random() for _ in range(5)]
        assert seq1 == seq2

    def test_seeding_numpy_is_deterministic(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not installed")

        set_seed(11)
        seq1 = np.random.rand(5).tolist()
        set_seed(11)
        seq2 = np.random.rand(5).tolist()
        assert seq1 == seq2

    def test_pythonhashseed_set_in_env(self):
        # We use setdefault, so this only fires if PYTHONHASHSEED is unset
        original = os.environ.pop("PYTHONHASHSEED", None)
        try:
            set_seed(42)
            assert os.environ["PYTHONHASHSEED"] == "42"
        finally:
            if original is not None:
                os.environ["PYTHONHASHSEED"] = original

    def test_report_as_dict_roundtrip(self):
        report = set_seed(99)
        d = report.as_dict()
        assert d["seed"] == 99
        assert {"python", "numpy", "mlx", "pythonhashseed"}.issubset(d.keys())


class TestGetSeedFromEnv:
    def test_uses_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("SIA_SEED", raising=False)
        assert get_seed_from_env() == DEFAULT_SEED

    def test_reads_env_int(self, monkeypatch):
        monkeypatch.setenv("SIA_SEED", "12345")
        assert get_seed_from_env() == 12345

    def test_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("SIA_SEED", "not-a-number")
        assert get_seed_from_env() == DEFAULT_SEED

    def test_default_override(self, monkeypatch):
        monkeypatch.delenv("SIA_SEED", raising=False)
        assert get_seed_from_env(default=999) == 999

    def test_empty_falls_back(self, monkeypatch):
        monkeypatch.setenv("SIA_SEED", "")
        assert get_seed_from_env() == DEFAULT_SEED
