import numpy as np
import pytest

from numcompute.utils import (
    clip_values,
    cosine_similarity,
    create_batches,
    ensure_2d,
    euclidean_distance,
    logsumexp,
    manhattan_distance,
    relu,
    sigmoid,
    softmax,
)


def test_euclidean_distance_simple():
    """Test basic Euclidean distance."""
    x1 = np.array([1.0, 2.0])
    x2 = np.array([4.0, 6.0])
    
    result = euclidean_distance(x1, x2)
    expected = np.sqrt(25 + 16)  # sqrt(41)
    assert np.isclose(result, expected)


def test_euclidean_distance_shape_mismatch_raises():
    """Test shape validation."""
    with pytest.raises(ValueError, match="same shape"):
        euclidean_distance([1, 2], [1, 2, 3])


def test_manhattan_distance_simple():
    """Test basic Manhattan distance."""
    x1 = np.array([1.0, 2.0])
    x2 = np.array([4.0, 6.0])
    
    result = manhattan_distance(x1, x2)
    assert np.isclose(result, 7.0)


def test_cosine_similarity_perfect():
    """Test perfect cosine similarity."""
    x1 = np.array([1.0, 0.0])
    x2 = np.array([1.0, 0.0])
    
    result = cosine_similarity(x1, x2)
    assert np.isclose(result, 1.0)


def test_cosine_similarity_zero_norm():
    """Test zero-norm handling."""
    result = cosine_similarity([1, 0], [0, 0])
    assert np.isclose(result, 0.0)


def test_sigmoid_basic():
    """Test sigmoid at key points."""
    x = np.array([-2.0, 0.0, 2.0])
    result = sigmoid(x)
    
    expected = np.array([0.11920292, 0.5, 0.88079708])
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_sigmoid_large_negative():
    """Test numerical stability for large negative values."""
    x = np.array([-100.0])
    result = sigmoid(x)
    assert result < 1e-40  # Should be effectively zero


def test_relu_basic():
    """Test ReLU activation."""
    x = np.array([-1.0, 0.0, 1.0])
    result = relu(x)
    
    expected = np.array([0.0, 0.0, 1.0])
    np.testing.assert_array_equal(result, expected)


def test_softmax_basic():
    """Test softmax normalization."""
    x = np.array([1.0, 2.0, 3.0])
    result = softmax(x)
    
    assert np.isclose(np.sum(result), 1.0, atol=1e-10)
    assert np.all(result > 0)


def test_softmax_axis_1():
    """Test softmax along specific axis."""
    x = np.array([[1, 2],
                  [3, 4]])
    result = softmax(x, axis=1)
    
    # Each row should sum to 1
    assert np.isclose(np.sum(result, axis=1), [1.0, 1.0]).all()


def test_softmax_numerical_stability():
    """Test stability with large values."""
    x = np.array([1000, 1001, 1002])
    result = softmax(x)
    
    assert np.isclose(np.sum(result), 1.0, atol=1e-10)
    assert np.all(result > 0)


def test_logsumexp_basic():
    """Test basic logsumexp computation."""
    x = np.array([1.0, 2.0, 3.0])
    result = logsumexp(x)
    
    expected = np.log(np.exp(1) + np.exp(2) + np.exp(3))
    assert np.isclose(result, expected)


def test_logsumexp_axis():
    """Test axis behavior."""
    x = np.array([[1, 2],
                  [3, 4]])
    result_axis0 = logsumexp(x, axis=0)
    result_axis1 = logsumexp(x, axis=1)
    
    assert result_axis0.shape == (2,)
    assert result_axis1.shape == (2,)


def test_logsumexp_numerical_stability():
    """Test stability with large values."""
    x = np.array([1000, 1001, 1002])
    result = logsumexp(x)
    
    # Should not overflow or underflow
    assert np.isfinite(result)


def test_create_batches_basic():
    """Test basic batching."""
    X = np.arange(10).reshape(-1, 1)
    batches = list(create_batches(X, batch_size=3))
    
    assert len(batches) == 4
    assert batches[0].shape == (3, 1)
    assert batches[3].shape == (1, 1)


def test_create_batches_with_y():
    """Test batching with targets."""
    X = np.arange(6).reshape(-1, 1)
    y = np.arange(6)
    
    batches = list(create_batches(X, y, batch_size=2))
    X_batch, y_batch = batches[0]
    
    assert X_batch.shape == (2, 1)
    assert y_batch.shape == (2,)


def test_create_batches_invalid_batch_size_raises():
    """Test invalid batch_size."""
    X = np.array([[1, 2]])
    
    with pytest.raises(ValueError, match="positive integer"):
        list(create_batches(X, batch_size=0))


def test_ensure_2d_1d_input():
    """Test 1D input conversion."""
    x = np.array([1, 2, 3])
    result = ensure_2d(x)
    
    assert result.shape == (3, 1)
    np.testing.assert_array_equal(result, [[1], [2], [3]])


def test_ensure_2d_2d_input():
    """Test 2D input passthrough."""
    x = np.array([[1, 2], [3, 4]])
    result = ensure_2d(x)
    
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result, x)


def test_ensure_2d_invalid_ndim_raises():
    """Test higher dimensional input."""
    with pytest.raises(ValueError, match="1D or 2D"):
        ensure_2d(np.array([[[1]]]));


def test_clip_values_basic():
    """Test basic clipping."""
    x = np.array([1.0, 3.0, 5.0])
    result = clip_values(x, min_val=2.0, max_val=4.0)
    
    expected = np.array([2.0, 3.0, 4.0])
    np.testing.assert_array_equal(result, expected)


def test_clip_values_no_clipping():
    """Test when values are within bounds."""
    x = np.array([1.0, 2.0, 3.0])
    result = clip_values(x, 0.0, 4.0)
    
    np.testing.assert_array_equal(result, x)


def test_softmax_all_equal():
    """Test softmax with identical values."""
    x = np.array([1.0, 1.0, 1.0])
    result = softmax(x)
    
    expected = np.array([1/3, 1/3, 1/3])
    np.testing.assert_allclose(result, expected, atol=1e-10)