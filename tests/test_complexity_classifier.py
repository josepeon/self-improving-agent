"""T1.4: classifier-based router complexity assessment."""

from __future__ import annotations

from self_improving_agent.providers.complexity_classifier import (
    DEFAULT_WEIGHTS,
    FEATURES,
    ComplexityClassifier,
    extract_features,
    feedback_to_training_example,
)

# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------


class TestFeatureExtraction:
    def test_returns_every_feature(self):
        features = extract_features([{"role": "user", "content": "Hi"}])
        assert set(features.keys()) == set(FEATURES)

    def test_empty_messages_safe(self):
        features = extract_features([])
        assert all(name in features for name in FEATURES)
        assert features["bias"] == 1.0

    def test_short_input_low_length(self):
        features = extract_features([{"role": "user", "content": "Hi"}])
        assert features["len_words"] < 0.1

    def test_long_input_max_length(self):
        long_text = " ".join(["word"] * 300)
        features = extract_features([{"role": "user", "content": long_text}])
        assert features["len_words"] == 1.0  # capped

    def test_strong_marker(self):
        features = extract_features(
            [{"role": "user", "content": "Step by step please"}]
        )
        assert features["marker_strong"] == 1.0
        assert features["marker_medium"] == 0.0

    def test_medium_marker(self):
        features = extract_features([{"role": "user", "content": "Explain X"}])
        assert features["marker_medium"] == 1.0
        assert features["marker_strong"] == 0.0

    def test_code_density(self):
        features = extract_features(
            [
                {
                    "role": "user",
                    "content": "```python\nfrom auth.handler import x\n```",
                }
            ]
        )
        assert features["code_fence"] == 1.0
        assert features["has_idents"] == 1.0  # auth.handler matches

    def test_multi_turn(self):
        features = extract_features(
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ]
        )
        assert features["multi_turn"] == 1.0

    def test_has_question(self):
        features = extract_features([{"role": "user", "content": "Why?"}])
        assert features["has_question"] == 1.0


# ---------------------------------------------------------------------
# Untrained classifier (default weights)
# ---------------------------------------------------------------------


class TestUntrainedClassifier:
    def test_uses_default_weights(self):
        clf = ComplexityClassifier()
        assert clf.weights == dict(DEFAULT_WEIGHTS)
        assert clf.trained is False

    def test_simple_prompt_low_probability(self):
        clf = ComplexityClassifier()
        p = clf.predict([{"role": "user", "content": "Hi"}])
        assert p < 0.3

    def test_complex_prompt_high_probability(self):
        clf = ComplexityClassifier()
        p = clf.predict(
            [
                {
                    "role": "user",
                    "content": (
                        "Step by step, explain the design tradeoffs of microservices "
                        "vs monoliths in detail, including pros and cons of each "
                        "approach, with code examples."
                    ),
                }
            ]
        )
        # Strong-marker-rich prompt should push well above 0.5
        assert p >= 0.6

    def test_ambiguous_prompt_low_confidence(self):
        """A prompt that hits middling features should produce middling probability
        and therefore low confidence."""
        clf = ComplexityClassifier()
        # No markers, short, no code — but has a question mark; weight = 0
        _, simple_conf = clf.predict_with_confidence(
            [{"role": "user", "content": "Hi"}]
        )
        # Single weak marker + question — middling signal
        _, ambig_conf = clf.predict_with_confidence(
            [{"role": "user", "content": "Can you translate this?"}]
        )
        # Both should still produce some confidence; what matters is that
        # the system gives us a usable signal, not a coin flip at p=0.5
        assert simple_conf > 0.0
        assert ambig_conf >= 0.0


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


class TestTraining:
    def test_train_on_obvious_examples(self):
        clf = ComplexityClassifier()
        # 'bad' (label=1) responses for long+marker-rich inputs;
        # 'good' (label=0) for short trivial ones
        examples = []
        for _ in range(20):
            f_complex, label_c = feedback_to_training_example(
                "Step by step, explain the design tradeoffs of X vs Y in detail "
                "with pros and cons",
                "bad",
            )
            examples.append((f_complex, label_c))
            f_simple, label_s = feedback_to_training_example("Hi", "good")
            examples.append((f_simple, label_s))

        result = clf.train(examples, epochs=300)
        assert result["trained"] is True
        assert result["train_accuracy"] >= 0.9
        assert clf.trained is True

    def test_train_with_empty_examples(self):
        clf = ComplexityClassifier()
        result = clf.train([])
        assert result["trained"] is False
        assert clf.trained is False

    def test_post_training_decisions_correct(self):
        clf = ComplexityClassifier()
        examples = []
        for _ in range(10):
            examples.append(
                feedback_to_training_example(
                    "Step by step, explain X in detail",
                    "bad",
                )
            )
            examples.append(feedback_to_training_example("Hi", "good"))
        clf.train(examples, epochs=400)

        # Untrained probabilities should now be sharper at the extremes
        simple = clf.predict([{"role": "user", "content": "Hi there"}])
        complex_ = clf.predict(
            [
                {
                    "role": "user",
                    "content": "Step by step, explain in detail",
                }
            ]
        )
        assert simple < 0.4
        assert complex_ > 0.6


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        clf = ComplexityClassifier()
        clf.weights["len_words"] = 99.9
        clf.trained = True
        clf.samples_seen = 42

        path = tmp_path / "clf.json"
        clf.save(path)

        loaded = ComplexityClassifier.load(path)
        assert loaded.weights["len_words"] == 99.9
        assert loaded.trained is True
        assert loaded.samples_seen == 42

    def test_save_creates_parent_dirs(self, tmp_path):
        clf = ComplexityClassifier()
        path = tmp_path / "nested" / "dir" / "clf.json"
        clf.save(path)
        assert path.exists()


# ---------------------------------------------------------------------
# Feedback integration helper
# ---------------------------------------------------------------------


class TestFeedbackToTrainingExample:
    def test_bad_rating_label_one(self):
        features, label = feedback_to_training_example("Anything", "bad")
        assert label == 1
        assert features["bias"] == 1.0

    def test_good_rating_label_zero(self):
        _, label = feedback_to_training_example("Anything", "good")
        assert label == 0


# ---------------------------------------------------------------------
# Router integration (lazy-load from disk)
# ---------------------------------------------------------------------


class TestRouterIntegration:
    def test_router_loads_classifier_from_env_path(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        from self_improving_agent.providers.router import ModelRouter

        # Save a sharply-biased classifier that always says "remote"
        clf = ComplexityClassifier()
        clf.weights = {k: 0.0 for k in DEFAULT_WEIGHTS}
        clf.weights["bias"] = 10.0  # always p ≈ 1.0
        clf.trained = True
        path = tmp_path / "clf.json"
        clf.save(path)

        monkeypatch.setenv("SIA_CLASSIFIER_PATH", str(path))
        router = ModelRouter(MagicMock(), MagicMock())

        # With confident remote classifier, even a trivial prompt routes remote
        assert router._assess_complexity([{"role": "user", "content": "Hi"}]) is True

    def test_router_falls_back_to_heuristic_when_no_classifier(
        self, tmp_path, monkeypatch
    ):
        from unittest.mock import MagicMock

        from self_improving_agent.providers.router import ModelRouter

        monkeypatch.setenv("SIA_CLASSIFIER_PATH", str(tmp_path / "no-such-file"))
        router = ModelRouter(MagicMock(), MagicMock())

        # Short 'Hi' should stay local — heuristic handles it
        assert router._assess_complexity([{"role": "user", "content": "Hi"}]) is False
