"""Active-learning feedback storage and integration.

Captures explicit user/operator feedback on production responses and
makes that signal available to the dataset builder.
"""

from self_improving_agent.feedback.store import (
    Feedback,
    FeedbackRating,
    FeedbackStore,
    feedback_to_sft_pair,
)

__all__ = [
    "Feedback",
    "FeedbackRating",
    "FeedbackStore",
    "feedback_to_sft_pair",
]
