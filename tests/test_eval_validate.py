"""B3: `sia eval-validate` — sanity check the eval pipeline without inference."""

from __future__ import annotations

import json

from self_improving_agent.cli import main


def _write_yaml(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestEvalValidate:
    def test_passes_for_valid_setup(self, tmp_path, capsys):
        suite = tmp_path / "tests"
        _write_yaml(
            suite / "core.yaml",
            (
                "tests:\n"
                "  - id: example\n"
                "    messages:\n"
                "      - role: user\n"
                "        content: hi\n"
            ),
        )
        result = main(
            [
                "eval-validate",
                "--test-suite",
                str(suite),
                "--registry",
                str(tmp_path / "registry.json"),
                "--reports-dir",
                str(tmp_path / "reports"),
            ]
        )
        out = capsys.readouterr().out
        assert result == 0
        assert "OK" in out
        assert "1 golden test" in out

    def test_fails_when_no_tests(self, tmp_path, capsys):
        result = main(
            [
                "eval-validate",
                "--test-suite",
                str(tmp_path / "empty"),
                "--registry",
                str(tmp_path / "registry.json"),
                "--reports-dir",
                str(tmp_path / "reports"),
            ]
        )
        out = capsys.readouterr().out
        assert result == 1
        assert "Only 0 golden test" in out

    def test_fails_for_test_missing_required_keys(self, tmp_path, capsys):
        suite = tmp_path / "tests"
        _write_yaml(
            suite / "broken.yaml",
            (
                "tests:\n  - id: bad\n"  # missing 'messages'
            ),
        )
        result = main(
            [
                "eval-validate",
                "--test-suite",
                str(suite),
                "--registry",
                str(tmp_path / "registry.json"),
                "--reports-dir",
                str(tmp_path / "reports"),
            ]
        )
        out = capsys.readouterr().out
        assert result == 1
        assert "missing keys" in out or "empty messages" in out

    def test_fails_for_corrupt_report(self, tmp_path, capsys):
        suite = tmp_path / "tests"
        _write_yaml(
            suite / "core.yaml",
            (
                "tests:\n"
                "  - id: ok\n"
                "    messages:\n"
                "      - role: user\n"
                "        content: hi\n"
            ),
        )
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "garbage.json").write_text("not json at all")

        result = main(
            [
                "eval-validate",
                "--test-suite",
                str(suite),
                "--registry",
                str(tmp_path / "registry.json"),
                "--reports-dir",
                str(reports),
            ]
        )
        out = capsys.readouterr().out
        assert result == 1
        assert "not valid JSON" in out

    def test_fails_for_report_missing_keys(self, tmp_path, capsys):
        suite = tmp_path / "tests"
        _write_yaml(
            suite / "core.yaml",
            (
                "tests:\n"
                "  - id: ok\n"
                "    messages:\n"
                "      - role: user\n"
                "        content: hi\n"
            ),
        )
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "partial.json").write_text(json.dumps({"model_version": "v1"}))

        result = main(
            [
                "eval-validate",
                "--test-suite",
                str(suite),
                "--registry",
                str(tmp_path / "registry.json"),
                "--reports-dir",
                str(reports),
            ]
        )
        out = capsys.readouterr().out
        assert result == 1
        assert "missing required keys" in out

    def test_min_tests_threshold(self, tmp_path, capsys):
        suite = tmp_path / "tests"
        _write_yaml(
            suite / "core.yaml",
            (
                "tests:\n"
                "  - id: ok\n"
                "    messages:\n"
                "      - role: user\n"
                "        content: hi\n"
            ),
        )
        # Only one test; require at least 5
        result = main(
            [
                "eval-validate",
                "--test-suite",
                str(suite),
                "--registry",
                str(tmp_path / "registry.json"),
                "--reports-dir",
                str(tmp_path / "reports"),
                "--min-tests",
                "5",
            ]
        )
        out = capsys.readouterr().out
        assert result == 1
        assert "minimum is 5" in out
