"""Tests for EvaluationHarness - golden test loading and execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from self_improving_agent.evaluation.harness import (
    EvaluationHarness,
    _check_latency,
    _check_max_length,
    _check_must_contain,
    _check_must_not_contain,
    _count_checks,
)

# ── Helpers ───────────────────────────────────────────────────────────


@dataclass
class FakeResponse:
    text: str = "Hello from OH companion!"


def _write_yaml(tmp_path: Path, tests: list[dict[str, Any]]) -> Path:
    """Write a minimal golden test YAML and return its directory."""
    suite = tmp_path / "golden"
    suite.mkdir()
    f = suite / "tests.yaml"
    f.write_text(yaml.dump({"tests": tests}))
    return suite


def _basic_test(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "id": "test_001",
        "category": "general",
        "description": "Basic test",
        "messages": [{"role": "user", "content": "Hello!"}],
        "expected": {
            "must_contain": [],
            "must_not_contain": ["error"],
        },
    }
    defaults.update(overrides)
    return defaults


# ── Check helpers (unit) ──────────────────────────────────────────────


class TestCheckHelpers:
    def test_must_contain_pass(self) -> None:
        assert _check_must_contain("Hello OH", {"must_contain": ["OH"]}) == []

    def test_must_contain_fail(self) -> None:
        result = _check_must_contain("Hello world", {"must_contain": ["OH"]})
        assert len(result) == 1
        assert "Missing" in result[0]

    def test_must_contain_case_insensitive(self) -> None:
        assert _check_must_contain("hello oh", {"must_contain": ["OH"]}) == []

    def test_must_not_contain_pass(self) -> None:
        assert _check_must_not_contain("Hello", {"must_not_contain": ["error"]}) == []

    def test_must_not_contain_fail(self) -> None:
        result = _check_must_not_contain("some error", {"must_not_contain": ["error"]})
        assert len(result) == 1
        assert "forbidden" in result[0].lower()

    def test_must_not_contain_case_insensitive(self) -> None:
        result = _check_must_not_contain("some ERROR", {"must_not_contain": ["error"]})
        assert len(result) == 1

    def test_latency_pass(self) -> None:
        assert _check_latency(100.0, {"response_time_ms": 200}) == []

    def test_latency_fail(self) -> None:
        result = _check_latency(300.0, {"response_time_ms": 200})
        assert len(result) == 1
        assert "slow" in result[0].lower()

    def test_latency_no_limit(self) -> None:
        assert _check_latency(99999.0, {}) == []

    def test_max_length_pass(self) -> None:
        assert _check_max_length("short", {"max_length": 100}) == []

    def test_max_length_fail(self) -> None:
        result = _check_max_length("x" * 200, {"max_length": 100})
        assert len(result) == 1
        assert "long" in result[0].lower()

    def test_max_length_no_limit(self) -> None:
        assert _check_max_length("x" * 10000, {}) == []

    def test_count_checks(self) -> None:
        expected = {
            "must_contain": ["a", "b"],
            "must_not_contain": ["c"],
            "response_time_ms": 1000,
            "max_length": 100,
        }
        assert _count_checks(expected) == 5

    def test_count_checks_empty(self) -> None:
        assert _count_checks({}) == 0


# ── Test loading ──────────────────────────────────────────────────────


class TestLoading:
    def test_loads_from_yaml(self, tmp_path: Path) -> None:
        suite_dir = _write_yaml(tmp_path, [_basic_test()])
        harness = EvaluationHarness(suite_dir)
        assert len(harness.tests) == 1

    def test_loads_multiple_tests(self, tmp_path: Path) -> None:
        tests = [_basic_test(id=f"t{i}") for i in range(5)]
        suite_dir = _write_yaml(tmp_path, tests)
        harness = EvaluationHarness(suite_dir)
        assert len(harness.tests) == 5

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        harness = EvaluationHarness(empty)
        assert harness.tests == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        harness = EvaluationHarness(tmp_path / "nope")
        assert harness.tests == []

    def test_loads_default_golden_tests(self) -> None:
        """The bundled conversations.yaml should load."""
        harness = EvaluationHarness()
        assert len(harness.tests) > 0


# ── run_test ──────────────────────────────────────────────────────────


class TestRunTest:
    async def test_passing_test(self, tmp_path: Path) -> None:
        test = _basic_test(
            expected={"must_contain": ["OH"], "must_not_contain": ["error"]}
        )
        suite_dir = _write_yaml(tmp_path, [test])
        harness = EvaluationHarness(suite_dir)

        async def gen(msg: str, *, history: list | None = None) -> FakeResponse:
            return FakeResponse(text="Hello from OH!")

        result = await harness.run_test(test, gen)
        assert result.passed
        assert result.score == 1.0
        assert result.latency_ms > 0
        assert result.failures == []

    async def test_failing_must_contain(self, tmp_path: Path) -> None:
        test = _basic_test(expected={"must_contain": ["OH"]})
        suite_dir = _write_yaml(tmp_path, [test])
        harness = EvaluationHarness(suite_dir)

        async def gen(msg: str, *, history: list | None = None) -> FakeResponse:
            return FakeResponse(text="Hello there!")

        result = await harness.run_test(test, gen)
        assert not result.passed
        assert result.score == 0.0

    async def test_failing_must_not_contain(self, tmp_path: Path) -> None:
        test = _basic_test(expected={"must_contain": [], "must_not_contain": ["error"]})
        suite_dir = _write_yaml(tmp_path, [test])
        harness = EvaluationHarness(suite_dir)

        async def gen(msg: str, *, history: list | None = None) -> FakeResponse:
            return FakeResponse(text="An error occurred")

        result = await harness.run_test(test, gen)
        assert not result.passed

    async def test_score_partial(self, tmp_path: Path) -> None:
        test = _basic_test(
            expected={"must_contain": ["OH", "companion"], "must_not_contain": []}
        )
        suite_dir = _write_yaml(tmp_path, [test])
        harness = EvaluationHarness(suite_dir)

        async def gen(msg: str, *, history: list | None = None) -> FakeResponse:
            return FakeResponse(text="Hello from OH!")  # missing "companion"

        result = await harness.run_test(test, gen)
        assert not result.passed
        assert result.score == 0.5

    async def test_string_response_fallback(self, tmp_path: Path) -> None:
        """When the response has no .text attr, str() is used."""
        test = _basic_test(expected={"must_contain": ["42"]})
        suite_dir = _write_yaml(tmp_path, [test])
        harness = EvaluationHarness(suite_dir)

        async def gen(msg: str, *, history: list | None = None) -> str:
            return "The answer is 42"

        result = await harness.run_test(test, gen)
        assert result.passed

    async def test_history_passed_to_generate(self, tmp_path: Path) -> None:
        test = _basic_test(
            messages=[
                {"role": "user", "content": "My name is Alex"},
                {"role": "assistant", "content": "Hi Alex!"},
                {"role": "user", "content": "What is my name?"},
            ],
            expected={"must_contain": ["Alex"]},
        )
        suite_dir = _write_yaml(tmp_path, [test])
        harness = EvaluationHarness(suite_dir)

        captured: dict[str, Any] = {}

        async def gen(msg: str, *, history: list | None = None) -> FakeResponse:
            captured["history"] = history
            return FakeResponse(text="Your name is Alex")

        await harness.run_test(test, gen)
        assert len(captured["history"]) == 2

    async def test_empty_expected(self, tmp_path: Path) -> None:
        test = _basic_test(expected={})
        suite_dir = _write_yaml(tmp_path, [test])
        harness = EvaluationHarness(suite_dir)

        async def gen(msg: str, *, history: list | None = None) -> FakeResponse:
            return FakeResponse()

        result = await harness.run_test(test, gen)
        assert result.passed
        assert result.score == 1.0


# ── run_all ───────────────────────────────────────────────────────────


class TestRunAll:
    async def test_report_fields(self, tmp_path: Path) -> None:
        tests = [_basic_test(id=f"t{i}", category="cat") for i in range(3)]
        suite_dir = _write_yaml(tmp_path, tests)
        harness = EvaluationHarness(suite_dir)

        async def gen(msg: str, *, history: list | None = None) -> FakeResponse:
            return FakeResponse()

        report = await harness.run_all(gen, "v1.0")
        assert report.model_version == "v1.0"
        assert report.total_tests == 3
        assert report.passed_tests == 3
        assert report.failed_tests == 0
        assert report.pass_rate == 1.0
        assert report.avg_latency_ms > 0
        assert "cat" in report.category_scores

    async def test_empty_suite(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        harness = EvaluationHarness(empty)

        async def gen(msg: str, *, history: list | None = None) -> FakeResponse:
            return FakeResponse()

        report = await harness.run_all(gen, "v0")
        assert report.total_tests == 0
        assert report.pass_rate == 0.0

    async def test_mixed_pass_fail(self, tmp_path: Path) -> None:
        tests = [
            _basic_test(id="pass", expected={"must_contain": ["OH"]}),
            _basic_test(id="fail", expected={"must_contain": ["MISSING"]}),
        ]
        suite_dir = _write_yaml(tmp_path, tests)
        harness = EvaluationHarness(suite_dir)

        async def gen(msg: str, *, history: list | None = None) -> FakeResponse:
            return FakeResponse(text="Hello from OH companion!")

        report = await harness.run_all(gen, "v1")
        assert report.passed_tests == 1
        assert report.failed_tests == 1
        assert report.pass_rate == 0.5


# ── run_category ──────────────────────────────────────────────────────


class TestRunCategory:
    async def test_filters_by_category(self, tmp_path: Path) -> None:
        tests = [
            _basic_test(id="g1", category="greeting"),
            _basic_test(id="s1", category="safety"),
            _basic_test(id="g2", category="greeting"),
        ]
        suite_dir = _write_yaml(tmp_path, tests)
        harness = EvaluationHarness(suite_dir)

        async def gen(msg: str, *, history: list | None = None) -> FakeResponse:
            return FakeResponse()

        report = await harness.run_category("greeting", gen, "v1")
        assert report.total_tests == 2
