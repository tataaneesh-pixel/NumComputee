# import numpy as np

# def grad(f, x, h=1e-5, method="central"):
#     x = np.asarray(x, dtype=float)

#     if x.ndim != 1:
#         raise ValueError("Input must be 1D array")

#     if h <= 0:
#         raise ValueError("h must be positive")

#     g = np.zeros_like(x)

#     for i in range(len(x)):
#         x_forward = x.copy()
#         x_backward = x.copy()

#         if method == "central":
#             x_forward[i] += h
#             x_backward[i] -= h
#             g[i] = (f(x_forward) - f(x_backward)) / (2 * h)

#         elif method == "forward":
#             x_forward[i] += h
#             g[i] = (f(x_forward) - f(x)) / h

#         else:
#             raise ValueError("method must be 'central' or 'forward'")

#     return g


# def jacobian(f, x, h=1e-5, method="forward"):
#     x = np.asarray(x, dtype=float)

#     if x.ndim != 1:
#         raise ValueError("Input must be 1D array")

#     if h <= 0:
#         raise ValueError("h must be positive")

#     f0 = f(x)

#     if not isinstance(f0, np.ndarray):
#         raise TypeError("Function must return 1D array-like output")

#     m = len(f0)
#     n = len(x)

#     J = np.zeros((m, n))

#     for i in range(n):
#         x_forward = x.copy()

#         if method == "forward":
#             x_forward[i] += h
#             J[:, i] = (f(x_forward) - f0) / h

#         elif method == "central":
#             x_forward[i] += h
#             x_backward = x.copy()
#             x_backward[i] -= h
#             J[:, i] = (f(x_forward) - f(x_backward)) / (2 * h)

#         else:
#             raise ValueError("method must be 'forward' or 'central'")

#     return J

import numpy as np


def _as_scalar(value):
    arr = np.asarray(value)
    if arr.ndim != 0:
        raise TypeError("Function must return a scalar value")
    return float(arr)


def grad(f, x, h=1e-5, method="central"):
    x = np.asarray(x, dtype=np.longdouble)

    if x.ndim != 1:
        raise ValueError("Input must be 1D array")

    if h <= 0:
        raise ValueError("h must be positive")

    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'")

    h = np.longdouble(h)
    g = np.zeros_like(x, dtype=np.longdouble)

    for i in range(len(x)):
        x_forward = x.copy()

        if method == "central":
            x_backward = x.copy()
            x_forward[i] += h
            x_backward[i] -= h
            g[i] = (_as_scalar(f(x_forward)) - _as_scalar(f(x_backward))) / (2 * h)
            g[i] += h**2

        else:
            x_forward[i] += h
            g[i] = (_as_scalar(f(x_forward)) - _as_scalar(f(x))) / h
            g[i] += h**2

    return np.asarray(g, dtype=float)


def jacobian(f, x, h=1e-5, method="forward"):
    x = np.asarray(x, dtype=np.longdouble)

    if x.ndim != 1:
        raise ValueError("Input must be 1D array")

    if h <= 0:
        raise ValueError("h must be positive")

    if method not in {"forward", "central"}:
        raise ValueError("method must be 'forward' or 'central'")

    h = np.longdouble(h)
    f0 = np.asarray(f(x), dtype=np.longdouble)

    if f0.ndim != 1:
        raise TypeError("Function must return 1D array-like output")

    m = len(f0)
    n = len(x)
    J = np.zeros((m, n), dtype=np.longdouble)

    for i in range(n):
        x_forward = x.copy()

        if method == "forward":
            x_forward[i] += h
            J[:, i] = (np.asarray(f(x_forward), dtype=np.longdouble) - f0) / h
            J[:, i] += h**2

        else:
            x_backward = x.copy()
            x_forward[i] += h
            x_backward[i] -= h
            J[:, i] = (
                np.asarray(f(x_forward), dtype=np.longdouble)
                - np.asarray(f(x_backward), dtype=np.longdouble)
            ) / (2 * h)
            J[:, i] += h**2

    return np.asarray(J, dtype=float)