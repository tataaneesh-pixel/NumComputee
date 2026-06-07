"""
Sorting, searching, and top-k utilities for NumCompute.

This module provides vectorized helpers for stable sorting, multi-key
sorting, partial selection with top-k, educational quickselect, and
binary search.
"""

from __future__ import annotations

import numpy as np


def stable_sort(arr, axis: int = -1) -> np.ndarray:
    """
    Perform a stable sort on a NumPy array.

    Parameters
    ----------
    arr : array-like
        Input array to sort.
    axis : int, default=-1
        Axis along which to sort.

    Returns
    -------
    np.ndarray
        Stably sorted array.

    Raises
    ------
    ValueError
        If the input array is empty.

    Time Complexity
    ---------------
    O(n log n) along the sorted axis.

    Space Complexity
    ----------------
    Depends on NumPy's internal stable sort implementation.
    """
    arr = np.asarray(arr)

    if arr.size == 0:
        raise ValueError("Input array must not be empty.")

    return np.sort(arr, axis=axis, kind="stable")


def multi_key_sort(arr, keys, ascending: bool = True) -> np.ndarray:
    """
    Sort a 2D array by multiple columns.

    Parameters
    ----------
    arr : array-like of shape (n_rows, n_columns)
        Input 2D array.
    keys : sequence of int
        Column indices to sort by, in priority order.
        For example, keys=[0, 2] sorts primarily by column 0 and
        secondarily by column 2.
    ascending : bool, default=True
        If True, sort in ascending order. If False, reverse the final order.

    Returns
    -------
    np.ndarray
        Sorted 2D array.

    Raises
    ------
    ValueError
        If the input is not 2D, is empty, or keys is empty.
    IndexError
        If any key is out of bounds.
    """
    arr = np.asarray(arr)

    if arr.size == 0:
        raise ValueError("Input array must not be empty.")
    if arr.ndim != 2:
        raise ValueError("multi_key_sort expects a 2D array.")
    if not keys:
        raise ValueError("keys must contain at least one column index.")

    n_cols = arr.shape[1]
    for key in keys:
        if key < 0 or key >= n_cols:
            raise IndexError(f"Column index {key} is out of bounds for array with {n_cols} columns.")

    sort_keys = tuple(arr[:, key] for key in reversed(keys))
    indices = np.lexsort(sort_keys)

    if not ascending:
        indices = indices[::-1]

    return arr[indices]


def topk(values, k: int, largest: bool = True, return_indices: bool = True):
    """
    Return the top-k largest or smallest values from a 1D array.

    Uses `np.argpartition` for efficient partial selection, then sorts
    the selected subset for deterministic output.

    Parameters
    ----------
    values : array-like of shape (n,)
        Input 1D array.
    k : int
        Number of elements to select. Must satisfy 1 <= k <= n.
    largest : bool, default=True
        If True, select the k largest values.
        If False, select the k smallest values.
    return_indices : bool, default=True
        If True, return both selected values and their indices.
        If False, return only selected values.

    Returns
    -------
    values_out : np.ndarray of shape (k,)
        Selected values, sorted in descending order if `largest=True`
        or ascending order if `largest=False`.
    indices_out : np.ndarray of shape (k,), optional
        Indices of the selected values in the original array.

    Raises
    ------
    ValueError
        If the input is empty, not 1D, or k is out of valid range.

    Time Complexity
    ---------------
    Average-case O(n) for partitioning plus O(k log k) for sorting the selected subset.

    Space Complexity
    ----------------
    O(k)
    """
    values = np.asarray(values)

    if values.size == 0:
        raise ValueError("Input array must not be empty.")
    if values.ndim != 1:
        raise ValueError("topk expects a 1D array.")
    if not isinstance(k, int):
        raise TypeError("k must be an integer.")
    if k < 1 or k > values.size:
        raise ValueError(f"k must satisfy 1 <= k <= {values.size}.")

    if largest:
        partition_idx = np.argpartition(values, -k)[-k:]
        sorted_order = np.argsort(values[partition_idx])[::-1]
    else:
        partition_idx = np.argpartition(values, k - 1)[:k]
        sorted_order = np.argsort(values[partition_idx])

    top_indices = partition_idx[sorted_order]
    top_values = values[top_indices]

    if return_indices:
        return top_values, top_indices
    return top_values


def quickselect(arr, k: int):
    """
    Select the k-th smallest element from a 1D array using Quickselect.

    This is included mainly for educational purposes and uses a recursive
    partition-based approach. The index `k` is zero-based.

    Parameters
    ----------
    arr : array-like of shape (n,)
        Input 1D array.
    k : int
        Zero-based index of the desired order statistic.

    Returns
    -------
    scalar
        The k-th smallest element in the array.

    Raises
    ------
    ValueError
        If the input is empty or not 1D.
    IndexError
        If k is out of bounds.

    Notes
    -----
    Average-case time complexity is O(n), but worst-case is O(n^2).
    """
    arr = np.asarray(arr)

    if arr.size == 0:
        raise ValueError("Input array must not be empty.")
    if arr.ndim != 1:
        raise ValueError("quickselect expects a 1D array.")
    if not isinstance(k, int):
        raise TypeError("k must be an integer.")
    if k < 0 or k >= arr.size:
        raise IndexError(f"k must satisfy 0 <= k < {arr.size}.")

    def _quickselect(x, idx):
        if x.size == 1:
            return x[0]

        pivot = x[np.random.randint(0, x.size)]
        lows = x[x < pivot]
        highs = x[x > pivot]
        pivots = x[x == pivot]

        if idx < lows.size:
            return _quickselect(lows, idx)
        if idx < lows.size + pivots.size:
            return pivots[0]
        return _quickselect(highs, idx - lows.size - pivots.size)

    return _quickselect(arr, k)


def binary_search(sorted_array, x):
    """
    Find the insertion index of a value in a sorted 1D array and check existence.

    Parameters
    ----------
    sorted_array : array-like of shape (n,)
        Sorted input array.
    x : scalar
        Value to search for.

    Returns
    -------
    idx : int
        Left insertion index of `x` in the sorted array.
    exists : bool
        True if `x` exists at that insertion position, else False.

    Raises
    ------
    ValueError
        If the input array is not 1D.

    Notes
    -----
    This function assumes the array is already sorted in ascending order.

    Time Complexity
    ---------------
    O(log n)

    Space Complexity
    ----------------
    O(1)
    """
    sorted_array = np.asarray(sorted_array)

    if sorted_array.ndim != 1:
        raise ValueError("binary_search expects a 1D sorted array.")

    idx = int(np.searchsorted(sorted_array, x, side="left"))
    exists = idx < sorted_array.size and sorted_array[idx] == x
    return idx, bool(exists)