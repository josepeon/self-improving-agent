"""
Classifier-based routing complexity assessment

The router upgrade in B2 replaced the keyword list with a hand-tuned
heuristic. That's defensible but it's still a fixed scoring function —
it doesn't *learn* from what happened to past requests.

This module is the next step: a small logistic regression that maps
extracted features to a probability "needs remote model". It can be
trained from the interaction log (interaction → was_satisfactory) so
routing decisions reflect actual outcomes, not hand-picked weights.

Without training data it falls back to hand-tuned default weights that
roughly reproduce the B2 heuristic, so the classifier is useful from
day one and improves as the interaction log accumulates.

This is deliberately simple — sklearn would handle this in 3 lines but
adds a heavy dep. Implemented in pure-Python with numpy for the math.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# Feature names — order matters; the weights vector is parallel to this list.
FEATURES = (
    "len_words",       # 0..1: words/200, capped
    "marker_strong",   # 0..1: has step-by-step/design/pros-and-cons
    "marker_medium",   # 0..1: has explain/analyze/why/compare
    "marker_weak",     # 0..1: has summarize/translate
    "code_fence",      # 0/1
    "has_paths",       # 0/1
    "has_idents",      # 0/1
    "multi_turn",      # 0/1: >2 messages
    "has_question",    # 0/1: contains a question mark
    "bias",            # always 1 (intercept term)
)


# Marker tier sets — only the words; the heuristic owns the weights.
_STRONG_MARKERS = frozenset({
    "step by step", "in detail", "pros and cons", "draft", "design",
})
_MEDIUM_MARKERS = frozenset({
    "explain", "analyze", "compare", "why", "how does",
    "debug", "evaluate", "write code", "what's the difference",
})
_WEAK_MARKERS = frozenset({
    "summarize", "translate", "calculate",
})

_CODE_FENCE_RE = re.compile(r"```")
_PATH_RE = re.compile(r"\b\w+/\w+")
_IDENT_RE = re.compile(
    r"\b(?:[a-z]+(?:_[a-z]+)+|[a-z]+[A-Z][a-zA-Z]+|\w+\.\w+)\b"
)


# Hand-tuned default weights that roughly reproduce the B2 heuristic in
# logistic-regression form. ``train()`` overwrites these once an
# interaction log is available.
DEFAULT_WEIGHTS = {
    "len_words": 1.2,
    "marker_strong": 2.0,
    "marker_medium": 1.0,
    "marker_weak": 0.4,
    "code_fence": 1.5,
    "has_paths": 0.6,
    "has_idents": 0.4,
    "multi_turn": 0.3,
    "has_question": 0.0,
    "bias": -2.4,
}


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------


def extract_features(messages: list[dict[str, str]]) -> dict[str, float]:
    """Pure feature extraction from chat messages.

    Returns a {feature_name: value} dict for each feature in FEATURES.
    Always returns every feature so the vector has a stable shape.
    """
    if not messages:
        return {name: 0.0 for name in FEATURES if name != "bias"} | {"bias": 1.0}

    user_msg = messages[-1].get("content", "")
    lowered = user_msg.lower()
    words = len(user_msg.split())

    return {
        "len_words": min(1.0, words / 200.0),
        "marker_strong": 1.0 if any(m in lowered for m in _STRONG_MARKERS) else 0.0,
        "marker_medium": 1.0 if any(m in lowered for m in _MEDIUM_MARKERS) else 0.0,
        "marker_weak": 1.0 if any(m in lowered for m in _WEAK_MARKERS) else 0.0,
        "code_fence": 1.0 if _CODE_FENCE_RE.search(user_msg) else 0.0,
        "has_paths": 1.0 if _PATH_RE.search(user_msg) else 0.0,
        "has_idents": 1.0 if _IDENT_RE.search(user_msg) else 0.0,
        "multi_turn": 1.0 if len(messages) > 2 else 0.0,
        "has_question": 1.0 if "?" in user_msg else 0.0,
        "bias": 1.0,
    }


# ---------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------


def _sigmoid(z: float) -> float:
    # Numerical stability — avoid math.exp overflow on extreme z
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class ComplexityClassifier:
    """Logistic regression on hand-engineered features.

    Untrained instance uses DEFAULT_WEIGHTS, which reproduce the B2
    heuristic in probabilistic form. ``train(examples)`` fits new
    weights from {features, label} examples.
    """

    weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS)
    )
    trained: bool = False
    samples_seen: int = 0

    # ------- inference -------

    def predict(self, messages: list[dict[str, str]]) -> float:
        """Return P(needs_remote) in [0, 1]."""
        features = extract_features(messages)
        z = sum(self.weights.get(k, 0.0) * v for k, v in features.items())
        return _sigmoid(z)

    def predict_with_confidence(
        self, messages: list[dict[str, str]]
    ) -> tuple[float, float]:
        """Return (probability, confidence).

        Confidence = |P - 0.5| * 2, scaled to [0, 1]. Used by the router
        to decide whether to trust the classifier or fall back to the
        heuristic.
        """
        p = self.predict(messages)
        return p, abs(p - 0.5) * 2

    # ------- training -------

    def train(
        self,
        examples: list[tuple[dict[str, float], int]],
        *,
        learning_rate: float = 0.05,
        epochs: int = 200,
    ) -> dict:
        """Fit weights by simple SGD on the supplied examples.

        Each example is (features_dict, label_int). Labels: 1 = needed
        remote (local gave a bad answer), 0 = local was fine. Returns
        the training summary dict.
        """
        if not examples:
            return {"trained": False, "reason": "no examples"}

        n = len(examples)
        # Initialise from current weights (carry forward learning)
        weights = dict(self.weights)

        for _ in range(epochs):
            # Single shuffle-free epoch — deterministic; reproducible
            for features, label in examples:
                z = sum(weights.get(k, 0.0) * v for k, v in features.items())
                prediction = _sigmoid(z)
                error = prediction - label
                # Per-feature gradient
                for k, v in features.items():
                    weights[k] -= learning_rate * error * v / n

        self.weights = weights
        self.trained = True
        self.samples_seen = n

        # Quick accuracy on the training set as a sanity check
        correct = sum(
            1 for features, label in examples
            if (_sigmoid(sum(weights.get(k, 0.0) * v for k, v in features.items())) >= 0.5)
            == bool(label)
        )
        return {
            "trained": True,
            "samples": n,
            "epochs": epochs,
            "train_accuracy": correct / n,
        }

    # ------- persistence -------

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            json.dump(
                {
                    "weights": self.weights,
                    "trained": self.trained,
                    "samples_seen": self.samples_seen,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> ComplexityClassifier:
        with open(path) as f:
            data = json.load(f)
        clf = cls(
            weights=dict(data.get("weights", DEFAULT_WEIGHTS)),
            trained=bool(data.get("trained", False)),
            samples_seen=int(data.get("samples_seen", 0)),
        )
        return clf


# ---------------------------------------------------------------------
# Training-data helpers
# ---------------------------------------------------------------------


def feedback_to_training_example(
    user_message: str,
    rating: str,
) -> tuple[dict[str, float], int]:
    """Convert one feedback row into a (features, label) tuple.

    Mapping: rating='bad' → 1 (the local model failed, should have gone
    remote). rating='good' → 0 (local was fine, keep routing it there).

    The feedback store (T2.2) records both. T1.4 trains on the union of
    those rows.
    """
    messages = [{"role": "user", "content": user_message}]
    features = extract_features(messages)
    label = 1 if rating == "bad" else 0
    return features, label
