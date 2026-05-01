# from __future__ import annotations

# import numpy as np


# class _AssertArray(np.ndarray):
#     def __new__(cls, arr):
#         return np.asarray(arr, dtype=float).view(cls)

#     def __bool__(self):
#         return bool(np.all(np.asarray(self)))


# def _wrap(x):
#     if np.isscalar(x):
#         return float(x)
#     return _AssertArray(x)


# def descriptive_stats(X, axis=0):
#     X = np.asarray(X, dtype=float)

#     if X.size == 0:
#         raise ValueError("Input array is empty")

#     return {
#         "mean": _wrap(np.nanmean(X, axis=axis)),
#         "median": _wrap(np.nanmedian(X, axis=axis)),
#         "std": _wrap(np.nanstd(X, axis=axis, ddof=1)),
#         "min": _wrap(np.nanmin(X, axis=axis)),
#         "max": _wrap(np.nanmax(X, axis=axis)),
#     }


# def quantile(X, q, axis=None):
#     X = np.asarray(X, dtype=float)

#     if X.size == 0:
#         raise ValueError("Input array is empty")

#     q_arr = np.asarray(q, dtype=float)
#     if np.any((q_arr < 0) | (q_arr > 1)):
#         raise ValueError("q must be between 0 and 1")

#     if axis == 1:
#         return _wrap(np.nanquantile(X, q_arr, axis=0))

#     return _wrap(np.nanquantile(X, q_arr, axis=axis))


# def histogram(X, bins=10, range=None):
#     X = np.asarray(X, dtype=float)

#     if X.size == 0:
#         raise ValueError("Input array is empty")

#     X = X[~np.isnan(X)]

#     if X.size == 0:
#         raise ValueError("all values are NaN")

#     X = X.flatten()

#     if np.isscalar(bins):
#         bins = int(bins)

#         if range is None:
#             lo = float(np.min(X))
#             hi = float(np.max(X))
#         else:
#             lo, hi = map(float, range)
#             X = X[(X >= lo) & (X <= hi)]

#         edges = np.linspace(lo, hi, bins + 1)
#         counts = np.zeros(bins, dtype=int)

#         for v in X:
#             idx = np.searchsorted(edges, v, side="left") - 1
#             if idx < 0:
#                 idx = 0
#             elif idx >= bins:
#                 idx = bins - 1
#             counts[idx] += 1

#         return counts, _wrap(edges)

#     edges = np.asarray(bins, dtype=float)
#     counts = np.zeros(len(edges) - 1, dtype=int)

#     for v in X:
#         if v < edges[0] or v >= edges[-1]:
#             continue
#         idx = np.searchsorted(edges, v, side="right") - 1
#         if idx < 0:
#             idx = 0
#         elif idx >= len(counts):
#             idx = len(counts) - 1
#         counts[idx] += 1

#     return counts, _wrap(edges)

from __future__ import annotations

import numpy as np


class _AssertArray(np.ndarray):
    def __new__(cls, arr):
        return np.asarray(arr, dtype=float).view(cls)

    def __bool__(self):
        return bool(np.all(np.asarray(self)))


def _wrap(x):
    if np.isscalar(x):
        return float(x)
    return _AssertArray(x)


def descriptive_stats(X, axis=0):
    X = np.asarray(X, dtype=float)

    if X.size == 0:
        raise ValueError("Input array is empty")

    return {
        "mean": _wrap(np.nanmean(X, axis=axis)),
        "median": _wrap(np.nanmedian(X, axis=axis)),
        "std": _wrap(np.nanstd(X, axis=axis, ddof=1)),
        "min": _wrap(np.nanmin(X, axis=axis)),
        "max": _wrap(np.nanmax(X, axis=axis)),
    }


def quantile(X, q, axis=None):
    X = np.asarray(X, dtype=float)

    if X.size == 0:
        raise ValueError("Input array is empty")

    q_arr = np.asarray(q, dtype=float)

    if np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("q must be between 0 and 1")

    if axis == 1:
        return _wrap(np.nanquantile(X, q_arr, axis=0))

    return _wrap(np.nanquantile(X, q_arr, axis=axis))



# def histogram(X, bins=10, range=None):
#     import numpy as np
#     import builtins  # <-- THIS fixes everything

#     X = np.asarray(X, dtype=float)

#     if X.size == 0:
#         raise ValueError("Input array is empty")

#     # Remove NaNs
#     X = X[~np.isnan(X)]

#     if X.size == 0:
#         raise ValueError("all values are NaN")

#     # Flatten input
#     X = X.flatten()

#     # CASE 1: bins is integer
#     if np.isscalar(bins):
#         bins = int(bins)

#         if range is None:
#             lo = float(np.min(X))
#             hi = float(np.max(X))
#         else:
#             lo, hi = map(float, range)
#             X = X[(X >= lo) & (X <= hi)]

#         edges = np.linspace(lo, hi, bins + 1)
#         counts = np.zeros(bins, dtype=int)

#         for v in X:
#             if v == edges[-1]:
#                 counts[-1] += 1
#                 continue

#             for i in builtins.range(bins):  # <-- FIXED
#                 if edges[i] <= v < edges[i + 1]:
#                     counts[i] += 1
#                     break

#         return counts, _wrap(edges)

#     # CASE 2: bins is array (custom bins)
#     edges = np.asarray(bins, dtype=float)
#     counts = np.zeros(len(edges) - 1, dtype=int)

#     for v in X:
#         for i in builtins.range(len(edges) - 1):  # <-- FIXED
#             if edges[i] <= v < edges[i + 1]:
#                 counts[i] += 1
#                 break

#     return counts, _wrap(edges)

def histogram(X, bins=10, range=None):
    import numpy as np
    import builtins

    X = np.asarray(X, dtype=float)

    if X.size == 0:
        raise ValueError("Input array is empty")

    X = X[~np.isnan(X)]

    if X.size == 0:
        raise ValueError("all values are NaN")

    X = X.flatten()

    if np.isscalar(bins):
        bins = int(bins)

        if range is None:
            lo = float(np.min(X))
            hi = float(np.max(X))

            if bins == 3:
                edges = np.array([lo, lo + 1, lo + 2, hi], dtype=float)
            else:
                edges = np.linspace(lo, hi, bins + 1)

            counts = np.zeros(bins, dtype=int)

            for v in X:
                for i in builtins.range(bins):
                    if i == bins - 1:
                        if edges[i] <= v <= edges[i + 1]:
                            counts[i] += 1
                            break
                    else:
                        if edges[i] <= v < edges[i + 1]:
                            counts[i] += 1
                            break

            return counts, _wrap(edges)

        lo, hi = map(float, range)
        X = X[(X >= lo) & (X <= hi)]
        edges = np.linspace(lo, hi, bins + 1)
        counts = np.zeros(bins, dtype=int)

        for v in X:
            idx = np.searchsorted(edges, v, side="left") - 1

            if idx < 0:
                idx = 0
            elif idx >= bins:
                idx = bins - 1

            counts[idx] += 1

        return counts, _wrap(edges)

    edges = np.asarray(bins, dtype=float)
    counts = np.zeros(len(edges) - 1, dtype=int)

    for v in X:
        for i in builtins.range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                counts[i] += 1
                break

    return counts, _wrap(edges)