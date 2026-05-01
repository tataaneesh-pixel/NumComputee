"""
Preprocessing utilities for NumCompute.

This module provides lightweight, NumPy-based preprocessing components
with a fit/transform/fit_transform API similar to common ML toolkits.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class Transformer:
    """
    Base class for preprocessors.

    Subclasses should implement `fit` and `transform`.
    """

    def fit(self, X):
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        self
            Fitted transformer.
        """
        return self

    def transform(self, X):
        """
        Transform the input data.

        Parameters
        ----------
        X : array-like
            Input data.

        Raises
        ------
        NotImplementedError
            If not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement transform().")

    def fit_transform(self, X):
        """
        Fit the transformer and return the transformed data.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        np.ndarray
            Transformed data.
        """
        return self.fit(X).transform(X)


def _ensure_2d_numeric(X: np.ndarray) -> np.ndarray:
    """
    Convert input to a 2D float64 NumPy array.

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    np.ndarray
        2D float64 array.

    Raises
    ------
    ValueError
        If the input is empty or has more than 2 dimensions.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.size == 0:
        raise ValueError("Input array must not be empty.")

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim != 2:
        raise ValueError("Input must be a 1D or 2D array.")

    return X


class StandardScaler(Transformer):
    """
    Standardize features using z-score normalization.

    For each feature:
        z = (x - mean) / std

    Constant columns are handled safely by replacing zero standard
    deviations with 1.0, so the transformed constant column becomes 0.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """
        Compute per-feature mean and standard deviation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Input numeric data.

        Returns
        -------
        self
            Fitted scaler.

        Raises
        ------
        ValueError
            If the input is empty or has invalid dimensions.

        Time Complexity
        ---------------
        O(n * m), where n is the number of samples and m is the number of features.

        Space Complexity
        ----------------
        O(m)
        """
        X = _ensure_2d_numeric(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """
        Standardize input data using fitted statistics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Input numeric data.

        Returns
        -------
        np.ndarray
            Standardized array of shape (n_samples, n_features).

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted.
        ValueError
            If the number of features does not match the fitted data.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler must be fitted before calling transform().")

        X = _ensure_2d_numeric(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        return (X - self.mean_) / self.std_


class MinMaxScaler(Transformer):
    """
    Scale features to a specified range.

    For each feature:
        x_scaled = (x - x_min) / (x_max - x_min)
        x_out = x_scaled * (high - low) + low

    Constant columns are handled safely by replacing zero ranges with 1.0.
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        low, high = feature_range
        if low >= high:
            raise ValueError("feature_range must satisfy low < high.")
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_range_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """
        Compute per-feature min and range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Input numeric data.

        Returns
        -------
        self
            Fitted scaler.

        Raises
        ------
        ValueError
            If the input is empty or has invalid dimensions.
        """
        X = _ensure_2d_numeric(X)
        data_min = np.min(X, axis=0)
        data_max = np.max(X, axis=0)
        data_range = data_max - data_min
        data_range = np.where(data_range == 0, 1.0, data_range)

        self.data_min_ = data_min
        self.data_range_ = data_range
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """
        Scale input data to the configured feature range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Input numeric data.

        Returns
        -------
        np.ndarray
            Scaled array of shape (n_samples, n_features).

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted.
        ValueError
            If the number of features does not match the fitted data.
        """
        if self.data_min_ is None or self.data_range_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before calling transform().")

        X = _ensure_2d_numeric(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        low, high = self.feature_range
        X_std = (X - self.data_min_) / self.data_range_
        return X_std * (high - low) + low


class OneHotEncoder(Transformer):
    """
    One-hot encode categorical features.

    Each input column is encoded independently, and the encoded columns
    are concatenated horizontally.

    Categories are learned during `fit` using `np.unique`, which returns
    sorted unique values.
    """

    def __init__(self):
        self.categories_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """
        Learn unique categories for each feature column.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Input categorical data.

        Returns
        -------
        self
            Fitted encoder.

        Raises
        ------
        ValueError
            If the input is empty or has more than 2 dimensions.
        """
        X = np.asarray(X)

        if X.size == 0:
            raise ValueError("Input array must not be empty.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError("Input must be a 1D or 2D array.")

        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """
        Transform categorical data into one-hot encoded numeric data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Input categorical data.

        Returns
        -------
        np.ndarray
            One-hot encoded array of shape
            (n_samples, sum(n_categories_per_feature)).

        Raises
        ------
        RuntimeError
            If the encoder has not been fitted.
        ValueError
            If the number of features does not match the fitted data.
        """
        if self.categories_ is None:
            raise RuntimeError("OneHotEncoder must be fitted before calling transform().")

        X = np.asarray(X)

        if X.size == 0:
            raise ValueError("Input array must not be empty.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError("Input must be a 1D or 2D array.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        encoded_columns = []
        for col_idx, categories in enumerate(self.categories_):
            column = X[:, col_idx][:, np.newaxis]
            encoded = (column == categories).astype(np.float64)
            encoded_columns.append(encoded)

        return np.hstack(encoded_columns)


class SimpleImputer(Transformer):
    """
    Replace missing values in numeric arrays.

    Supported strategies:
    - 'mean': replace NaN values with column-wise means
    - 'constant': replace NaN values with a fixed constant
    """

    def __init__(self, strategy: str = "mean", fill_value: float = 0.0):
        if strategy not in {"mean", "constant"}:
            raise ValueError("strategy must be either 'mean' or 'constant'.")
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """
        Compute replacement values for missing data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Input numeric data containing possible NaN values.

        Returns
        -------
        self
            Fitted imputer.

        Raises
        ------
        ValueError
            If the input is empty or has invalid dimensions.
        """
        X = _ensure_2d_numeric(X)

        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        else:
            self.statistics_ = np.full(X.shape[1], self.fill_value, dtype=np.float64)

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """
        Replace NaN values using fitted statistics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Input numeric data containing possible NaN values.

        Returns
        -------
        np.ndarray
            Imputed numeric array of shape (n_samples, n_features).

        Raises
        ------
        RuntimeError
            If the imputer has not been fitted.
        ValueError
            If the number of features does not match the fitted data.
        """
        if self.statistics_ is None:
            raise RuntimeError("SimpleImputer must be fitted before calling transform().")

        X = _ensure_2d_numeric(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        X_out = X.copy()
        mask = np.isnan(X_out)

        if np.any(mask):
            row_idx, col_idx = np.where(mask)
            X_out[row_idx, col_idx] = self.statistics_[col_idx]

        return X_out