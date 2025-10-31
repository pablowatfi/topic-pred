"""Central default settings for the topic_pred package.

Keep default values in a single place so they are easy to change.
"""
from __future__ import annotations

DEFAULT_TOP_K: int = 3
DEFAULT_THRESHOLD: float = 0.3

# Default artifacts dir and filename (relative to project root)
DEFAULT_ARTIFACTS_DIR: str = "artifacts"
DEFAULT_ARTIFACT_FILENAME: str = "topic_predictor_direct_model.pkl"
