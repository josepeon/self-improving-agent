"""Tests for the CLI entry point."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from self_improving_agent.cli import build_parser, main
from self_improving_agent.training.registry import (
    ModelRegistry,
    ModelStatus,
    ModelVersion,
)


# ===========================================
# Parser smoke
# ===========================================

class TestParser:
    def test_subcommands_registered(self):
        parser = build_parser()
        # Argparse stores subparser names on the _subparsers_action choices dict
        actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
        assert actions, "no subparsers found"
        names = set(actions[0].choices.keys())
        assert {"status", "evaluate", "train", "serve"}.issubset(names)


# ===========================================
# Process-level
# ===========================================

class TestCLIProcess:
    def test_version_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "self_improving_agent.cli", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_no_command_shows_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "self_improving_agent.cli"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "Self-Improving" in result.stdout

    def test_help_lists_each_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "self_improving_agent.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        for cmd in ["evaluate", "train", "serve", "status"]:
            assert cmd in result.stdout


# ===========================================
# sia status — purely file-driven, safe to exercise in-process
# ===========================================

class TestStatusCommand:
    def test_empty_registry_returns_zero(self, tmp_path):
        registry_path = tmp_path / "registry.json"
        result = main(["status", "--registry", str(registry_path)])
        assert result == 0

    def test_renders_table_for_registry(self, tmp_path, capsys):
        registry_path = tmp_path / "registry.json"
        registry = ModelRegistry(registry_path)
        registry.register(ModelVersion(
            version="v1.0",
            base_model="meta-llama/llama-3-8b",
            adapter_path="models/finetuned/v1.0/adapters",
            status=ModelStatus.PRODUCTION,
        ))
        registry.register(ModelVersion(
            version="v1.1",
            base_model="meta-llama/llama-3-8b",
            status=ModelStatus.CANARY,
        ))

        result = main([
            "status",
            "--registry", str(registry_path),
            "--reports-dir", str(tmp_path / "reports"),
        ])
        out = capsys.readouterr().out
        assert result == 0
        assert "v1.0" in out
        assert "v1.1" in out
        assert "production" in out.lower()

    def test_json_output_is_machine_parseable(self, tmp_path, capsys):
        registry_path = tmp_path / "registry.json"
        registry = ModelRegistry(registry_path)
        registry.register(ModelVersion(
            version="v1.0", base_model="m", status=ModelStatus.PRODUCTION,
        ))
        main([
            "status", "--registry", str(registry_path), "--json",
            "--reports-dir", str(tmp_path / "reports"),
        ])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert any(row["version"] == "v1.0" for row in data)

    def test_attaches_latest_report(self, tmp_path, capsys):
        registry_path = tmp_path / "registry.json"
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        registry = ModelRegistry(registry_path)
        registry.register(ModelVersion(
            version="v1.0", base_model="m", status=ModelStatus.PRODUCTION,
        ))
        # A matching report file
        (reports_dir / "v1.0_1700000000.json").write_text(json.dumps({
            "pass_rate": 0.87,
            "passed_tests": 87,
            "total_tests": 100,
        }))
        main([
            "status", "--registry", str(registry_path),
            "--reports-dir", str(reports_dir), "--json",
        ])
        data = json.loads(capsys.readouterr().out)
        assert data[0]["latest_report"]["pass_rate"] == 0.87


# ===========================================
# sia evaluate — exercise resolution logic without touching MLX
# ===========================================

class TestEvaluateCommand:
    def test_no_target_returns_exit_code_2(self, tmp_path):
        registry_path = tmp_path / "registry.json"
        result = main([
            "evaluate",
            "--registry", str(registry_path),
            "--reports-dir", str(tmp_path / "reports"),
            "--test-suite", str(tmp_path / "tests"),
        ])
        assert result == 2

    def test_unknown_version_raises_systemexit(self, tmp_path):
        registry_path = tmp_path / "registry.json"
        # Empty registry; passing --version that doesn't exist
        with pytest.raises(SystemExit):
            main([
                "evaluate",
                "--version", "v9.9-not-real",
                "--registry", str(registry_path),
                "--reports-dir", str(tmp_path / "reports"),
            ])

    def test_remote_without_tests_returns_2(self, tmp_path):
        # --remote bypasses registry, but if there are no golden tests the
        # command should still exit with 2 and print a clear message.
        result = main([
            "evaluate",
            "--remote",
            "--registry", str(tmp_path / "registry.json"),
            "--reports-dir", str(tmp_path / "reports"),
            "--test-suite", str(tmp_path / "no_tests_here"),
        ])
        assert result == 2


# ===========================================
# sia train — dry-run is safe and the main behavior we want covered
# ===========================================

class TestTrainCommand:
    def test_dry_run_with_no_logs(self, tmp_path, capsys):
        # Empty interaction log -> dry-run should report 0 entries cleanly
        registry_path = tmp_path / "registry.json"
        result = main([
            "train", "--dry-run",
            "--registry", str(registry_path),
            "--log-base-path", str(tmp_path / "logs"),
            "--reports-dir", str(tmp_path / "reports"),
            "--finetuned-dir", str(tmp_path / "finetuned"),
        ])
        out = capsys.readouterr().out
        assert result in (0, 1)  # log read may surface as 1 if dir missing
        assert "Interaction log:" in out or "Could not read" in out
