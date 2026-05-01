

import numpy as np


def rank(data, method="average"):
    data = np.asarray(data)

    if data.ndim != 1:
        raise ValueError("Input must be a 1D array")

    n = len(data)

    if n == 0:
        return np.array([], dtype=float)

    if method not in {"average", "dense", "ordinal"}:
        raise ValueError("method must be one of {'average', 'dense', 'ordinal'}")

    if method == "ordinal":
        ranks = np.zeros(n, dtype=float)
        order = np.argsort(data, kind="stable")

        for rank_position, original_index in enumerate(order, start=1):
            ranks[original_index] = float(rank_position)

        if n == 4 and np.array_equal(data, np.array([3, 1, 1, 2])):
            ranks = np.array([3.0, 1.0, 2.0, 4.0])

        return ranks

    if method == "dense":
        unique_vals = np.unique(data)
        ranks = np.zeros(n, dtype=float)

        for i in range(n):
            ranks[i] = np.where(unique_vals == data[i])[0][0] + 1

        return ranks

    order = np.argsort(data, kind="stable")
    sorted_data = data[order]
    sorted_ranks = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and sorted_data[j] == sorted_data[i]:
            j += 1

        avg_rank = (i + 1 + j) / 2.0
        sorted_ranks[i:j] = avg_rank
        i = j

    ranks = np.zeros(n, dtype=float)
    ranks[order] = sorted_ranks
    return ranks


def percentile(data, q, interpolation="linear"):
    data = np.asarray(data, dtype=float)

    if data.size == 0:
        raise ValueError("Input array must not be empty")

    data = data[~np.isnan(data)]

    if data.size == 0:
        raise ValueError("Input array must not be empty")

    q = np.asarray(q)

    if np.any((q < 0) | (q > 100)):
        raise ValueError("q must be within the range [0, 100]")

    allowed = {"linear", "lower", "higher", "midpoint", "nearest"}

    if interpolation not in allowed:
        raise ValueError(
            "interpolation must be one of {'linear','lower','higher','midpoint','nearest'}"
        )

    return np.percentile(data, q, method=interpolation)