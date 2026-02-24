"""Evaluation harness - golden test suites and automated quality gates."""

from self_improving_agent.evaluation.harness import EvaluationHarness
from self_improving_agent.evaluation.models import EvaluationReport, TestResult

__all__ = ["EvaluationHarness", "EvaluationReport", "TestResult"]
