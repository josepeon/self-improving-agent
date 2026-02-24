"""Tests for the CLI entry point."""

import subprocess
import sys


class TestCLI:
    """Tests for the CLI module."""

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

    def test_unimplemented_command_exits_with_error(self):
        result = subprocess.run(
            [sys.executable, "-m", "self_improving_agent.cli", "evaluate"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "not yet implemented" in result.stdout

    def test_available_commands(self):
        result = subprocess.run(
            [sys.executable, "-m", "self_improving_agent.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        for cmd in ["evaluate", "train", "serve", "status"]:
            assert cmd in result.stdout
