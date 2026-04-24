import numpy as np
def euclidean_distance(x1, x2):
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    return np.sum(np.abs(x1 - x2))

def cosine_similarity(x1, x2):
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    if norm_x1 == 0 or norm_x2 == 0:
        return 0.0
    
    return dot_product / (norm_x1 * norm_x2)

def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    output = np.zeros_like(x)
    positive_mask = x >= 0
    negative_mask = ~positive_mask
    output[positive_mask] = 1 / (1 + np.exp(-x[positive_mask]))
    exp_x = np.exp(x[negative_mask])
    output[negative_mask] = exp_x / (1 + exp_x)
    
    return output

def relu(x):
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(0, x)

def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def logsumexp(x, axis=None):
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=axis, keepdims=True)
    result = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    
    return result
def create_batches(X, y=None, batch_size=32):
    n_samples = X.shape[0]
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        
        X_batch = X[start_idx:end_idx]
        
        if y is not None:
            y_batch = y[start_idx:end_idx]
            yield X_batch, y_batch
        else:
            yield X_batch
def ensure_2d(X):
    X = np.asarray(X)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    return X
def clip_values(x, min_val=None, max_val=None):
    return np.clip(x, min_val, max_val)