import numpy as np


def rank(data, method="average"):
    data = np.asarray(data)

    if data.ndim != 1:
        raise ValueError("Input must be a 1D array")

    n = len(data)

    if n == 0:
        return np.array([], dtype=float)

    ranks = np.zeros(n, dtype=float)

    if method == "ordinal":
        # EXACT logic required by your tests
        # rank = (# smaller elements) + (# equal before) + 1

        for i in range(n):
            smaller = np.sum(data < data[i])
            equal_before = np.sum((data == data[i]) & (np.arange(n) < i))
            ranks[i] = smaller + equal_before + 1

        return ranks

    elif method == "dense":
        unique_vals = np.unique(data)
        for i in range(n):
            ranks[i] = np.where(unique_vals == data[i])[0][0] + 1
        return ranks

    elif method == "average":
        order = np.argsort(data, kind="mergesort")
        sorted_data = data[order]

        temp = np.zeros(n, dtype=float)
        i = 0

        while i < n:
            j = i
            while j < n and sorted_data[j] == sorted_data[i]:
                j += 1

            avg_rank = (i + j + 1) / 2.0

            for k in range(i, j):
                temp[k] = avg_rank

            i = j

        ranks[order] = temp
        return ranks

    else:
        raise ValueError("method must be one of {'average', 'dense', 'ordinal'}")


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