"""Evaluation harness — loads golden tests, runs them, produces reports.

In addition to the original static rubrics (must_contain, must_not_contain,
response_time_ms, max_length), tests can now opt into an LLM-as-judge
rubric. In a golden YAML:

    - id: friendly_tone
      messages:
        - role: user
          content: Hi
      expected:
        judge:
          model: groq
          criterion: "Response is friendly and welcoming."
          threshold: 0.7   # 0..1, default 0.7

The judge prompts the LLM with the user message, the model's response,
and the criterion; expects a 0..1 score back. Below the threshold counts
as a failure on the standard checks list.

The judge is only invoked when ``judge`` is present in the test's
``expected`` block — tests without it cost zero extra inference. The
provider is constructed lazily so the harness stays importable in CI
without API keys.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import yaml

from self_improving_agent.evaluation.models import EvaluationReport, TestResult

logger = logging.getLogger(__name__)

# Type alias for the async generate function injected at runtime.
# Signature: generate(message, *, history=[]) -> response
GenerateFn = Callable[..., Coroutine[Any, Any, Any]]

_DEFAULT_SUITE = Path(__file__).parent / "golden_tests"


class EvaluationHarness:
    """Load golden tests and evaluate any ``generate_fn`` against them.

    Parameters
    ----------
    test_suite_path:
        Directory containing ``*.yaml`` golden test files.
    """

    def __init__(self, test_suite_path: str | Path | None = None) -> None:
        self.test_suite_path = (
            Path(test_suite_path) if test_suite_path else _DEFAULT_SUITE
        )
        self.tests: list[dict[str, Any]] = self._load_tests()

    # ── Loading ───────────────────────────────────────────────────────

    def _load_tests(self) -> list[dict[str, Any]]:
        """Read all ``*.yaml`` files in the suite directory."""
        tests: list[dict[str, Any]] = []
        if not self.test_suite_path.exists():
            return tests
        for yaml_file in sorted(self.test_suite_path.glob("*.yaml")):
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            if data and "tests" in data:
                tests.extend(data["tests"])
        return tests

    # ── Single test ───────────────────────────────────────────────────

    async def run_test(
        self,
        test: dict[str, Any],
        generate_fn: GenerateFn,
    ) -> TestResult:
        """Run one golden test case and return its result."""
        failures: list[str] = []

        messages = test["messages"]
        user_message = messages[-1]["content"]
        history = messages[:-1] if len(messages) > 1 else []

        start = time.perf_counter()
        response = await generate_fn(user_message, history=history)
        latency_ms = (time.perf_counter() - start) * 1000

        response_text: str = getattr(response, "text", str(response))
        expected = test.get("expected", {})

        # ── Checks ────────────────────────────────────────
        failures.extend(_check_must_contain(response_text, expected))
        failures.extend(_check_must_not_contain(response_text, expected))
        failures.extend(_check_latency(latency_ms, expected))
        failures.extend(_check_max_length(response_text, expected))
        failures.extend(await _check_judge(user_message, response_text, expected))

        total_checks = _count_checks(expected)
        passed_checks = total_checks - len(failures)
        score = passed_checks / total_checks if total_checks > 0 else 1.0

        return TestResult(
            test_id=test["id"],
            passed=len(failures) == 0,
            score=score,
            latency_ms=latency_ms,
            response=response_text,
            failures=failures,
        )

    # ── Full suite ────────────────────────────────────────────────────

    async def run_all(
        self,
        generate_fn: GenerateFn,
        model_version: str,
    ) -> EvaluationReport:
        """Run every golden test and produce an :class:`EvaluationReport`."""
        results: list[TestResult] = []
        category_buckets: dict[str, list[float]] = {}

        for test in self.tests:
            result = await self.run_test(test, generate_fn)
            results.append(result)

            cat = test.get("category", "general")
            category_buckets.setdefault(cat, []).append(result.score)

        category_scores = {
            cat: sum(scores) / len(scores) for cat, scores in category_buckets.items()
        }

        passed = sum(1 for r in results if r.passed)
        total = len(results)

        return EvaluationReport(
            model_version=model_version,
            total_tests=total,
            passed_tests=passed,
            failed_tests=total - passed,
            pass_rate=passed / total if total else 0.0,
            avg_latency_ms=(
                sum(r.latency_ms for r in results) / total if total else 0.0
            ),
            category_scores=category_scores,
            results=results,
        )

    # ── Filtered runs ─────────────────────────────────────────────────

    async def run_category(
        self,
        category: str,
        generate_fn: GenerateFn,
        model_version: str,
    ) -> EvaluationReport:
        """Run only tests in *category*."""
        filtered = [t for t in self.tests if t.get("category") == category]
        harness = EvaluationHarness.__new__(EvaluationHarness)
        harness.test_suite_path = self.test_suite_path
        harness.tests = filtered
        return await harness.run_all(generate_fn, model_version)


# ── Private check helpers ─────────────────────────────────────────────


def _check_must_contain(text: str, expected: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    lower = text.lower()
    for phrase in expected.get("must_contain", []):
        if phrase.lower() not in lower:
            failures.append(f"Missing required phrase: '{phrase}'")
    return failures


def _check_must_not_contain(text: str, expected: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    lower = text.lower()
    for phrase in expected.get("must_not_contain", []):
        if phrase.lower() in lower:
            failures.append(f"Contains forbidden phrase: '{phrase}'")
    return failures


def _check_latency(latency_ms: float, expected: dict[str, Any]) -> list[str]:
    max_latency = expected.get("response_time_ms")
    if max_latency is not None and latency_ms > max_latency:
        return [f"Too slow: {latency_ms:.0f}ms > {max_latency}ms"]
    return []


def _check_max_length(text: str, expected: dict[str, Any]) -> list[str]:
    max_length = expected.get("max_length")
    if max_length is not None and len(text) > max_length:
        return [f"Too long: {len(text)} > {max_length} chars"]
    return []


def _count_checks(expected: dict[str, Any]) -> int:
    """Count the total number of individual checks for score calculation."""
    return (
        len(expected.get("must_contain", []))
        + len(expected.get("must_not_contain", []))
        + (1 if "response_time_ms" in expected else 0)
        + (1 if "max_length" in expected else 0)
        + (1 if "judge" in expected else 0)
    )


# ──────────────────────────────────────────────────────────────────────
# LLM-as-judge rubric
# ──────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are a careful evaluator. Given a user message, an assistant "
    "response, and a criterion, score how well the response satisfies "
    "the criterion on a scale of 0.0 to 1.0. Respond with only the number."
)

_JUDGE_SCORE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\b")


def _parse_judge_score(text: str) -> float | None:
    """Pull a 0..1 score out of the judge's response. Returns None on failure."""
    match = _JUDGE_SCORE_RE.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if value > 1.0:
        # Some judges return 0..10 — normalise gracefully
        value = value / 10.0
    return max(0.0, min(1.0, value))


# Cache provider instances per (provider_name, model_name) so each call
# doesn't pay the constructor cost. Wiped on test runs via tests' monkeypatch.
_provider_cache: dict[tuple[str, str | None], Any] = {}


def _get_judge_provider(provider_name: str, model: str | None) -> Any:
    cache_key = (provider_name, model)
    if cache_key in _provider_cache:
        return _provider_cache[cache_key]

    if provider_name == "groq":
        from self_improving_agent.providers.groq_provider import GroqProvider

        provider = GroqProvider(model=model)
    elif provider_name == "mlx":
        from self_improving_agent.providers.mlx_provider import MLXProvider

        provider = MLXProvider(model_path=model)
    else:
        raise ValueError(
            f"Unknown judge provider {provider_name!r}; choose 'groq' or 'mlx'."
        )
    _provider_cache[cache_key] = provider
    return provider


def clear_judge_cache() -> None:
    """Reset the cached judge providers — used by tests."""
    _provider_cache.clear()


async def _check_judge(
    user_message: str, response_text: str, expected: dict[str, Any]
) -> list[str]:
    """Run the LLM-as-judge rubric if the test asked for one."""
    judge_spec = expected.get("judge")
    if not judge_spec:
        return []

    if not isinstance(judge_spec, dict):
        return [f"Invalid judge spec: expected dict, got {type(judge_spec).__name__}"]

    criterion = judge_spec.get("criterion")
    if not criterion:
        return ["Judge rubric missing 'criterion' field"]

    provider_name = judge_spec.get("model", "groq")
    model = judge_spec.get("model_name")  # optional specific model id
    threshold = float(judge_spec.get("threshold", 0.7))

    try:
        provider = _get_judge_provider(provider_name, model)
    except Exception as exc:
        return [f"Could not build judge provider: {exc}"]

    from self_improving_agent.providers.base import GenerationConfig

    judge_prompt = (
        f"User message:\n{user_message}\n\n"
        f"Assistant response:\n{response_text}\n\n"
        f"Criterion: {criterion}\n\n"
        f"Score (0.0 to 1.0):"
    )

    try:
        response = await provider.generate(
            [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": judge_prompt},
            ],
            GenerationConfig(temperature=0.2, max_tokens=32),
        )
    except Exception as exc:
        logger.warning("Judge provider failed: %s", exc)
        return [f"Judge unavailable: {type(exc).__name__}"]

    score = _parse_judge_score(response.text)
    if score is None:
        return [f"Judge returned unparseable score: {response.text[:80]!r}"]

    if score < threshold:
        return [
            f"Judge score {score:.2f} below threshold {threshold:.2f} "
            f"on criterion: {criterion}"
        ]
    return []
