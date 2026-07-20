"""T3.3: scripts/_lib utility helpers."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _lib import (
    ensure_dir,
    init_logging,
    load_env_or_die,
    make_progress,
    read_jsonl,
    resolve_path,
    write_jsonl,
)


class TestInitLogging:
    def test_idempotent(self, caplog):
        init_logging(verbose=False)
        init_logging(verbose=False)
        # No exception; root logger isn't doubly configured

    def test_verbose_sets_debug(self):
        # Reset to allow fresh configuration
        logging.getLogger().handlers.clear()
        init_logging(verbose=True)
        assert logging.getLogger().level == logging.DEBUG


class TestLoadEnvOrDie:
    def test_returns_existing_keys(self, monkeypatch):
        monkeypatch.setenv("FOO", "x")
        monkeypatch.setenv("BAR", "y")
        result = load_env_or_die("FOO", "BAR")
        assert result == {"FOO": "x", "BAR": "y"}

    def test_exits_on_missing(self, monkeypatch, capsys):
        monkeypatch.delenv("NONEXISTENT_KEY_FOR_TEST", raising=False)
        with pytest.raises(SystemExit):
            load_env_or_die("NONEXISTENT_KEY_FOR_TEST")
        captured = capsys.readouterr()
        assert "Missing required" in captured.err

    def test_default_groq_api_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "stub")
        result = load_env_or_die()
        assert "GROQ_API_KEY" in result

    def test_empty_value_counted_missing(self, monkeypatch):
        monkeypatch.setenv("EMPTY_KEY", "")
        with pytest.raises(SystemExit):
            load_env_or_die("EMPTY_KEY")


class TestResolvePath:
    def test_expands_tilde(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        p = resolve_path("~/file.txt")
        assert str(p).startswith(str(tmp_path))

    def test_returns_absolute(self, tmp_path):
        relative = "relative-path"
        result = resolve_path(relative)
        assert result.is_absolute()

    def test_must_exist_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_path(tmp_path / "no-such-file", must_exist=True)

    def test_must_exist_accepts(self, tmp_path):
        target = tmp_path / "exists.txt"
        target.write_text("x")
        assert resolve_path(target, must_exist=True) == target.resolve()


class TestEnsureDir:
    def test_creates(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        result = ensure_dir(nested)
        assert nested.exists()
        assert result == nested.resolve()

    def test_idempotent(self, tmp_path):
        ensure_dir(tmp_path)
        ensure_dir(tmp_path)  # no exception on second call


class TestMakeProgress:
    def test_yields_all_items(self):
        items = list(make_progress(range(5), total=5, label="test"))
        assert items == [0, 1, 2, 3, 4]

    def test_empty_iterable(self):
        items = list(make_progress([], total=0))
        assert items == []


class TestJsonlIO:
    def test_read_write_round_trip(self, tmp_path):
        path = tmp_path / "rows.jsonl"
        rows = [{"a": 1}, {"b": 2}, {"nested": {"x": [1, 2]}}]
        count = write_jsonl(path, rows)
        assert count == 3
        reloaded = list(read_jsonl(path))
        assert reloaded == rows

    def test_read_skips_malformed(self, tmp_path, caplog):
        path = tmp_path / "mixed.jsonl"
        path.write_text(
            json.dumps({"good": 1})
            + "\n"
            + "not json\n"
            + json.dumps({"good": 2})
            + "\n"
        )
        with caplog.at_level(logging.WARNING):
            rows = list(read_jsonl(path))
        assert rows == [{"good": 1}, {"good": 2}]
        assert any("malformed" in r.message for r in caplog.records)

    def test_write_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "deeply" / "out.jsonl"
        count = write_jsonl(path, [{"a": 1}])
        assert count == 1
        assert path.exists()

    def test_read_blank_lines_skipped(self, tmp_path):
        path = tmp_path / "blanks.jsonl"
        path.write_text(json.dumps({"a": 1}) + "\n\n\n" + json.dumps({"a": 2}) + "\n")
        assert list(read_jsonl(path)) == [{"a": 1}, {"a": 2}]

    def test_read_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(read_jsonl(tmp_path / "no-such-file"))
