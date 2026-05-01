"""
NumCompute: A lightweight scientific computing and ML utility toolkit
built with plain Python and NumPy.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .io import load_csv
from .metrics import accuracy, auc, confusion_matrix, f1, mse, precision, recall, roc_curve
from .optim import grad, jacobian
from .pipeline import Pipeline
from .preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler
from .rank import percentile, rank
from .sort_search import binary_search, multi_key_sort, quickselect, stable_sort, topk
from .stats import descriptive_stats, histogram, quantile
from .utils import (
    clip_values,
    cosine_similarity,
    create_batches,
    ensure_2d,
    euclidean_distance,
    logsumexp,
    manhattan_distance,
    relu,
    sigmoid,
    softmax,
)

__all__ = [
    "load_csv",
    "StandardScaler",
    "MinMaxScaler",
    "OneHotEncoder",
    "SimpleImputer",
    "stable_sort",
    "multi_key_sort",
    "topk",
    "quickselect",
    "binary_search",
    "rank",
    "percentile",
    "descriptive_stats",
    "quantile",
    "histogram",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "mse",
    "roc_curve",
    "auc",
    "grad",
    "jacobian",
    "Pipeline",
    "euclidean_distance",
    "manhattan_distance",
    "cosine_similarity",
    "sigmoid",
    "relu",
    "softmax",
    "logsumexp",
    "create_batches",
    "ensure_2d",
    "clip_values",
]