"""Embedding helpers using sentence-transformers.

Expose a single model loader and an `encode_texts` helper so all parts of
the package share the same model instance and loading logic.
"""
from __future__ import annotations

from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import re
from collections import Counter

_MODEL_CACHE = {}

def get_model(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> SentenceTransformer:
    """Return a cached SentenceTransformer instance for `model_name`.

    The instance is created lazily and reused to avoid repeated downloads.
    """
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def encode_texts(texts: str, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> np.ndarray:
    model = get_model(model_name)
    arr = model.encode([texts], batch_size=32)
    return np.array(arr)

def preproc_content_texts(title: str | None, description: str | None) -> str:
  if title is None and description is None:
      return "There is not enough information to describe the content. Title and Descritpion are both missing."
  return ((title or '') + ' ' + (description or '')).strip()

