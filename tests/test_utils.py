import numpy as np
import pytest
from numcompute.utils import (
    euclidean_distance, manhattan_distance, cosine_similarity,
    sigmoid, relu, softmax, logsumexp,
    create_batches, ensure_2d, clip_values
)

def test_euclidean_distance():
    d = euclidean_distance([0, 0], [3, 4])
    assert np.isclose(d, 5.0), f"Expected 5.0, got {d}"
    d = euclidean_distance([1, 2, 3], [1, 2, 3])
    assert np.isclose(d, 0.0), "Distance to self shld be 0"
    print("Euclidean distance works!")


def test_manhattan_distance():
    d = manhattan_distance([0, 0], [3, 4])
    assert np.isclose(d, 7.0), f"Expected 7.0, got {d}"
    d = manhattan_distance([1, 2, 3], [1, 2, 3])
    assert np.isclose(d, 0.0), "Distance to self shld be 0"
    print("Manhattan distance works!")


def test_cosine_similarity():
    sim = cosine_similarity([1, 0, 0], [1, 0, 0])
    assert np.isclose(sim, 1.0), "Identical vectors shld have similarity 1"
    sim = cosine_similarity([1, 0], [0, 1])
    assert np.isclose(sim, 0.0), "Orthogonal vectors shld have similarity 0"
    sim = cosine_similarity([1, 0], [-1, 0])
    assert np.isclose(sim, -1.0), "Opposite vectors shld have similarity -1"
    
    print("Cosine similarity works!")


def test_sigmoid():
    result = sigmoid(0)
    assert np.isclose(result, 0.5), f"sigmoid(0) shld be 0.5, got {result}"
    result = sigmoid(10)
    assert result > 0.99, "sigmoid(10) shld be close to 1"
    result = sigmoid(-10)
    assert result < 0.01, "sigmoid(-10) shld be close to 0"
    print("Sigmoid works!")


def test_relu():
    x = np.array([-2, -1, 0, 1, 2])
    result = relu(x)
    expected = np.array([0, 0, 0, 1, 2])
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"
    print("ReLU works!")


def test_softmax():
    x = np.array([1.0, 2.0, 3.0])
    result = softmax(x)
    assert np.isclose(result.sum(), 1.0), "Softmax shld sum to 1"
    assert np.all(result > 0), "All softmax values shld be positive"
    assert np.argmax(result) == 2, "Largest input shld give largest probability"
    
    print("Softmax works!")


def test_logsumexp():
    x = np.array([1.0, 2.0, 3.0])
    result = logsumexp(x)
    naive = np.log(np.sum(np.exp(x)))
    assert np.isclose(result, naive), "logsumexp shld match naive computation"
    x_large = np.array([1000, 1001, 1002])
    result_large = logsumexp(x_large)
    assert not np.isinf(result_large), "logsumexp shld not overflow on large values"
    
    print("logsumexp works!")


def test_create_batches():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])
    
    batches = list(create_batches(X, y, batch_size=2))
    assert len(batches) == 3, f"Expected 3 batches, got {len(batches)}"
    X_batch, y_batch = batches[0]
    assert X_batch.shape == (2, 2), f"Expected shape (2, 2), got {X_batch.shape}"
    assert y_batch.shape == (2,), f"Expected shape (2,), got {y_batch.shape}"
    X_batch, y_batch = batches[2]
    assert X_batch.shape == (1, 2), f"Expected shape (1, 2), got {X_batch.shape}"
    
    print("create_batches works!")


def test_ensure_2d():
    x = np.array([1, 2, 3])
    result = ensure_2d(x)
    assert result.shape == (3, 1), f"Expected shape (3, 1), got {result.shape}"
    x = np.array([[1, 2], [3, 4]])
    result = ensure_2d(x)
    assert result.shape == (2, 2), f"Expected shape (2, 2), got {result.shape}"
    print("ensure_2d works!")


def test_clip_values():
    x = np.array([1, 5, 10, 15, 20])
    result = clip_values(x, min_val=5, max_val=15)
    expected = np.array([5, 5, 10, 15, 15])
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"
    print("clip_values works!")