"""
CSV I/O utilities for loading data into NumPy arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def load_csv(
    filepath,
    delimiter: str = ",",
    skip_header: bool = True,
    missing_values="",
    filling_values=np.nan,
    dtype=float,
    usecols=None,
) -> np.ndarray:
    """
    Load a CSV or delimited text file into a NumPy array.

    Parameters
    ----------
    filepath : str or Path
        Path to the file.
    delimiter : str, default=","
        Delimiter used in the file.
    skip_header : bool, default=True
        Whether to skip the first header row.
    missing_values : str or iterable, default=""
        Tokens to treat as missing values.
    filling_values : scalar, default=np.nan
        Value to use for missing entries.
    dtype : data-type, default=float
        Output dtype passed to NumPy.
    usecols : int or sequence of int, optional
        Column index or indices to read.

    Returns
    -------
    np.ndarray
        Loaded data as a NumPy array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty or cannot be read.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        data = np.genfromtxt(
            path,
            delimiter=delimiter,
            skip_header=1 if skip_header else 0,
            missing_values=missing_values,
            filling_values=filling_values,
            dtype=dtype,
            usecols=usecols,
            ndmin=0,
        )
    except Exception as exc:
        raise ValueError(f"Failed to load file '{path}': {exc}") from exc

    if data.size == 0:
        raise ValueError(f"File '{path}' is empty or contains no readable data.")

    if np.isscalar(data) or getattr(data, "ndim", None) == 0:
        data = np.array([[data]], dtype=dtype)
    elif data.ndim == 1:
        if usecols is None:
            data = data.reshape(1, -1)
        else:
            data = data.reshape(-1, 1)

    return data