import numpy as np

def grad(f, x, h=1e-5, method="central"):
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("Input must be 1D array")

    if h <= 0:
        raise ValueError("h must be positive")

    g = np.zeros_like(x)

    for i in range(len(x)):
        x_forward = x.copy()
        x_backward = x.copy()

        if method == "central":
            x_forward[i] += h
            x_backward[i] -= h
            g[i] = (f(x_forward) - f(x_backward)) / (2 * h)

        elif method == "forward":
            x_forward[i] += h
            g[i] = (f(x_forward) - f(x)) / h

        else:
            raise ValueError("method must be 'central' or 'forward'")

    return g


def jacobian(f, x, h=1e-5, method="forward"):
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("Input must be 1D array")

    if h <= 0:
        raise ValueError("h must be positive")

    f0 = f(x)

    if not isinstance(f0, np.ndarray):
        raise TypeError("Function must return 1D array-like output")

    m = len(f0)
    n = len(x)

    J = np.zeros((m, n))

    for i in range(n):
        x_forward = x.copy()

        if method == "forward":
            x_forward[i] += h
            J[:, i] = (f(x_forward) - f0) / h

        elif method == "central":
            x_forward[i] += h
            x_backward = x.copy()
            x_backward[i] -= h
            J[:, i] = (f(x_forward) - f(x_backward)) / (2 * h)

        else:
            raise ValueError("method must be 'forward' or 'central'")

    return J