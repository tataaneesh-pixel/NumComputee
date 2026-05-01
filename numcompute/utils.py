
import numpy as np




def euclidean_distance(x1, x2) -> float:
    """
    Compute Euclidean distance between two vectors.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    if x1.shape != x2.shape:
        raise ValueError("same shape")

    return float(np.sqrt(np.sum((x2 - x1) ** 2)))


def manhattan_distance(x1, x2) -> float:
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    if x1.shape != x2.shape:
        raise ValueError("x1 and x2 must have the same shape.")

    return float(np.sum(np.abs(x1 - x2)))


def cosine_similarity(x1, x2) -> float:
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    if x1.shape != x2.shape:
        raise ValueError("x1 and x2 must have the same shape.")

    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)

    if norm_x1 == 0.0 or norm_x2 == 0.0:
        return 0.0

    return float(np.dot(x1, x2) / (norm_x1 * norm_x2))


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)

    positive = x >= 0
    negative = ~positive

    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[negative])
    out[negative] = exp_x / (1.0 + exp_x)

    return out


def relu(x):
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(0.0, x)


def softmax(x, axis: int = -1):
    x = np.asarray(x, dtype=np.float64)

    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    denom = np.sum(exp_x, axis=axis, keepdims=True)

    return exp_x / denom


def logsumexp(x, axis=None):
    x = np.asarray(x, dtype=np.float64)

    x_max = np.max(x, axis=axis, keepdims=True)
    result = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))

    if axis is None:
        return float(np.squeeze(result))

    return np.squeeze(result, axis=axis)


def create_batches(X, y=None, batch_size: int = 32):
    X = np.asarray(X)

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    if X.ndim == 0:
        raise ValueError("X must have at least one dimension.")

    n_samples = X.shape[0]

    if y is not None:
        y = np.asarray(y)

        if y.shape[0] != n_samples:
            raise ValueError("X and y must have the same number of samples.")

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        X_batch = X[start:end]

        if y is not None:
            y_batch = y[start:end]
            yield X_batch, y_batch
        else:
            yield X_batch


def ensure_2d(X):
    X = np.asarray(X)

    if X.ndim == 1:
        return X.reshape(-1, 1)

    if X.ndim == 2:
        return X

    raise ValueError("Input must be a 1D or 2D array.")


def clip_values(x, min_val=None, max_val=None):
    x = np.asarray(x, dtype=np.float64)
    return np.clip(x, min_val, max_val)