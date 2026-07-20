"""Intelligent model router — routes requests to the optimal provider.

Routing is heuristic (not a classifier). The previous version did
``any(keyword in user_msg for keyword in COMPLEXITY_MARKERS)`` — that's
honest if you call it a keyword list, dishonest if you call it complexity
analysis. We now use a small score made of three signals:

  1. Length: long prompts (>~200 tokens of message content) usually need
     a more capable model.
  2. Reasoning markers: presence of words/phrases that historically
     correlate with hard requests ("why", "compare", "step by step",
     "draft", "design"). Same list as before but applied with weight.
  3. Code / technical density: presence of code fences, file paths, or
     identifier-like tokens (snake_case, camelCase, dot-separated names).

A request scores 0..1. Threshold for routing remote defaults to 0.5;
override via constructor arg or COMPLEXITY_THRESHOLD env. T1.4 will
replace this with a classifier trained on the interaction log.
"""

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from self_improving_agent.providers.base import (
    BaseModelProvider,
    GenerationConfig,
    ProviderResponse,
)

if TYPE_CHECKING:
    from self_improving_agent.providers.complexity_classifier import (
        ComplexityClassifier,
    )

logger = logging.getLogger(__name__)


_CODE_FENCE_RE = re.compile(r"```")
_PATH_RE = re.compile(r"\b\w+/\w+", re.UNICODE)
_IDENT_RE = re.compile(r"\b(?:[a-z]+(?:_[a-z]+)+|[a-z]+[A-Z][a-zA-Z]+|\w+\.\w+)\b")


def _word_count(text: str) -> int:
    return len(text.split())


class ModelRouter:
    """Routes requests to the optimal provider based on a heuristic score.

    Routing logic (priority):
      1. Explicit ``force_provider`` overrides everything.
      2. ``quality_mode`` "fast" → local, "quality" → remote.
      3. Heuristic complexity score ≥ threshold → remote.
      4. Default: local with automatic fallback on failure.

    The router tracks request counts + fallback rate so the dashboard
    (T1.2) can show actual routing behaviour.
    """

    # Reasoning-style phrases, weighted.
    REASONING_MARKERS: ClassVar[dict[str, float]] = {
        "why": 0.25,
        "compare": 0.25,
        "explain": 0.2,
        "analyze": 0.25,
        "step by step": 0.4,
        "in detail": 0.3,
        "pros and cons": 0.35,
        "design": 0.3,
        "draft": 0.3,
        "summarize": 0.2,
        "translate": 0.15,
        "evaluate": 0.25,
        "how does": 0.25,
        "what's the difference": 0.25,
        "write code": 0.3,
        "debug": 0.3,
        "calculate": 0.2,
    }

    # Backward-compatible alias — exposed as a flat list of marker strings.
    # Old callers / tests can still iterate ``COMPLEXITY_MARKERS``.
    COMPLEXITY_MARKERS: ClassVar[list[str]] = list(REASONING_MARKERS.keys())

    # When the heuristic isn't sure, fall back to local. This is the safer
    # default since local is free and fast.
    DEFAULT_THRESHOLD: ClassVar[float] = 0.5

    # Wordcount above which we start adding weight just from length alone.
    LENGTH_FLOOR: ClassVar[int] = 60
    LENGTH_CAP: ClassVar[int] = 300  # length contribution maxes out here

    def __init__(
        self,
        local: BaseModelProvider,
        remote: BaseModelProvider,
        complexity_threshold: float | None = None,
        classifier: "ComplexityClassifier | None" = None,
    ):
        self.local = local
        self.remote = remote

        # Performance tracking
        self._local_requests = 0
        self._remote_requests = 0
        self._fallback_count = 0
        self._total_requests = 0
        self._classifier_used = 0

        # Heuristic threshold; env override for live tuning without code change
        raw_env = os.environ.get("COMPLEXITY_THRESHOLD", "").strip()
        if complexity_threshold is not None:
            self._threshold = complexity_threshold
        elif raw_env:
            try:
                self._threshold = float(raw_env)
            except ValueError:
                logger.warning(
                    "COMPLEXITY_THRESHOLD=%r is not a valid float; using default %.2f",
                    raw_env,
                    self.DEFAULT_THRESHOLD,
                )
                self._threshold = self.DEFAULT_THRESHOLD
        else:
            self._threshold = self.DEFAULT_THRESHOLD

        # Optional classifier — soft signal. If unset and a trained
        # classifier exists at SIA_CLASSIFIER_PATH, load it lazily.
        self._classifier = classifier
        self._classifier_attempted_load = False

    @property
    def stats(self) -> dict:
        """Current routing statistics."""
        return {
            "total_requests": self._total_requests,
            "local_requests": self._local_requests,
            "remote_requests": self._remote_requests,
            "fallback_count": self._fallback_count,
            "fallback_rate": (
                self._fallback_count / self._total_requests
                if self._total_requests > 0
                else 0.0
            ),
            "local_ratio": (
                self._local_requests / self._total_requests
                if self._total_requests > 0
                else 0.0
            ),
        }

    def _complexity_score(self, messages: list[dict[str, str]]) -> float:
        """Return a 0..1 score; ≥ threshold means "send to remote."

        Signal weights:
          - Length: gradual ramp from LENGTH_FLOOR to LENGTH_CAP words.
          - Reasoning markers: sum of per-marker weights (capped at 0.6).
          - Technical density: code fences (+0.3), paths (+0.1 per up to 0.2),
            identifier-like tokens (+0.05 per up to 0.15).
          - Multi-turn: presence of >2 messages adds +0.1 (more context to
            keep track of).

        The cap of 1.0 is enforced once at the end.
        """
        if not messages:
            return 0.0

        user_msg = messages[-1].get("content", "")
        lowered = user_msg.lower()

        # Length signal
        words = _word_count(user_msg)
        if words <= self.LENGTH_FLOOR:
            length_score = 0.0
        elif words >= self.LENGTH_CAP:
            length_score = 0.3
        else:
            length_score = 0.3 * (
                (words - self.LENGTH_FLOOR) / (self.LENGTH_CAP - self.LENGTH_FLOOR)
            )

        # Reasoning markers
        marker_score = 0.0
        for marker, weight in self.REASONING_MARKERS.items():
            if marker in lowered:
                marker_score += weight
        marker_score = min(marker_score, 0.6)

        # Technical density
        tech_score = 0.0
        if _CODE_FENCE_RE.search(user_msg):
            tech_score += 0.3
        path_hits = len(_PATH_RE.findall(user_msg))
        tech_score += min(path_hits * 0.1, 0.2)
        ident_hits = len(_IDENT_RE.findall(user_msg))
        tech_score += min(ident_hits * 0.05, 0.15)

        # Multi-turn context
        multi_turn_score = 0.1 if len(messages) > 2 else 0.0

        total = length_score + marker_score + tech_score + multi_turn_score
        return min(total, 1.0)

    def _ensure_classifier(self):
        """Lazy-load a trained classifier from disk if available."""
        if self._classifier is not None or self._classifier_attempted_load:
            return
        self._classifier_attempted_load = True
        path = os.environ.get(
            "SIA_CLASSIFIER_PATH", "models/complexity_classifier.json"
        )
        if Path(path).exists():
            try:
                from self_improving_agent.providers.complexity_classifier import (
                    ComplexityClassifier,
                )

                self._classifier = ComplexityClassifier.load(path)
                logger.info("Loaded complexity classifier from %s", path)
            except Exception as exc:
                logger.warning("Could not load classifier at %s: %s", path, exc)

    def _assess_complexity(self, messages: list[dict[str, str]]) -> bool:
        """Decide whether the request should route remote.

        Strategy:
          1. If a classifier is available and confident (|p - 0.5| * 2 >= 0.4),
             trust its prediction.
          2. Otherwise fall back to the hand-tuned heuristic score.
        """
        self._ensure_classifier()
        if self._classifier is not None:
            p, confidence = self._classifier.predict_with_confidence(messages)
            if confidence >= 0.4:
                self._classifier_used += 1
                return p >= 0.5
        return self._complexity_score(messages) >= self._threshold

    async def route(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
        quality_mode: str = "default",
        force_provider: str | None = None,
    ) -> ProviderResponse:
        """Route a request to the optimal provider.

        Args:
            messages: Chat messages in OpenAI format.
            config: Generation parameters.
            quality_mode: "fast" (local), "quality" (remote), or "default" (auto).
            force_provider: Override routing with "local" or "remote".

        Returns:
            ProviderResponse from the selected provider.
        """
        self._total_requests += 1

        # 1. Explicit override
        if force_provider == "remote":
            logger.debug("Forced routing to remote provider")
            self._remote_requests += 1
            return await self.remote.generate(messages, config)
        if force_provider == "local":
            logger.debug("Forced routing to local provider")
            self._local_requests += 1
            return await self.local.generate(messages, config)

        # 2. Quality mode
        if quality_mode == "fast":
            logger.debug("Quality mode 'fast' → local provider")
            self._local_requests += 1
            return await self._try_local_with_fallback(messages, config)
        if quality_mode == "quality":
            logger.debug("Quality mode 'quality' → remote provider")
            self._remote_requests += 1
            return await self.remote.generate(messages, config)

        # 3. Complexity-based routing
        if self._assess_complexity(messages):
            logger.debug("Complex request detected → remote provider")
            self._remote_requests += 1
            return await self.remote.generate(messages, config)

        # 4. Default: local with fallback
        logger.debug("Default routing → local provider with fallback")
        self._local_requests += 1
        return await self._try_local_with_fallback(messages, config)

    async def _try_local_with_fallback(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> ProviderResponse:
        """Try local provider, fall back to remote on failure.

        Args:
            messages: Chat messages.
            config: Generation parameters.

        Returns:
            ProviderResponse from whichever provider succeeds.
        """
        try:
            return await self.local.generate(messages, config)
        except Exception as e:
            logger.warning(
                "Local provider failed (%s), falling back to remote: %s",
                self.local.name,
                e,
            )
            self._fallback_count += 1
            response = await self.remote.generate(messages, config)
            response.is_fallback = True
            return response

    async def route_stream(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
        quality_mode: str = "default",
        force_provider: str | None = None,
    ):
        """Route a streaming request to the optimal provider.

        Same routing logic as route(), but returns an async generator.
        """
        self._total_requests += 1

        # Determine provider
        if force_provider == "remote":
            provider = self.remote
        elif force_provider == "local":
            provider = self.local
        elif quality_mode == "quality":
            provider = self.remote
        elif quality_mode == "fast":
            provider = self.local
        elif self._assess_complexity(messages):
            provider = self.remote
        else:
            provider = self.local

        if provider == self.local:
            self._local_requests += 1
        else:
            self._remote_requests += 1

        try:
            async for token in provider.generate_stream(messages, config):
                yield token
        except Exception as e:
            if provider == self.local:
                logger.warning("Local streaming failed, falling back to remote: %s", e)
                self._fallback_count += 1
                async for token in self.remote.generate_stream(messages, config):
                    yield token
            else:
                raise
