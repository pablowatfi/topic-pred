"""Utilities for topic_pred package.

Expose IO and embedding helpers.
"""
from .io import load_csv
from .preprocess_methods import get_model, encode_texts

__all__ = ["load_csv", "get_model", "encode_texts"]
