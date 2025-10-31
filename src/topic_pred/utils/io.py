"""Data IO helpers for topic_pred.

Provides a single function `load_data` that reads the project's CSV files
from a data directory. This keeps file loading consistent across modules.
"""
from __future__ import annotations

import os
from typing import Tuple

import pandas as pd


def load_csv(file_name: str, data_dir: str = "data") -> pd.DataFrame:
    """Load a single CSV from `data_dir`.

    Example usage:
        load_csv('content.csv')
        load_csv('topics.csv')
        load_csv('correlations.csv')

    Raises FileNotFoundError with a helpful message if the requested file
    does not exist under `data_dir`.
    """
    path = os.path.join(data_dir, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_csv(path)
