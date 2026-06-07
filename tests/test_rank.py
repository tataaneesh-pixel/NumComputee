import numpy as np
import pytest

from numcompute.rank import percentile, rank


def test_rank_average_ties():
    """Test average ranking with ties."""
    data = np.array([10, 20, 20, 30])
    
    result = rank(data, method="average")
    
    expected = np.array([1.0, 2.5, 2.5, 4.0])
    np.testing.assert_allclose(result, expected)


def test_rank_dense_ties():
    """Test dense ranking with ties."""
    data = np.array([10, 20, 20, 30])
    
    result = rank(data, method="dense")
    
    expected = np.array([1.0, 2.0, 2.0, 3.0])
    np.testing.assert_allclose(result, expected)


def test_rank_ordinal_ties():
    """Test ordinal ranking with ties."""
    data = np.array([10, 20, 20, 30])
    
    result = rank(data, method="ordinal")
    
    expected = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(result, expected)


def test_rank_all_identical_values():
    """Test ranking when all values are the same."""
    data = np.array([5, 5, 5, 5])
    
    result_avg = rank(data, method="average")
    result_dense = rank(data, method="dense")
    
    expected_avg = np.full(4, 2.5)
    expected_dense = np.full(4, 1.0)
    
    np.testing.assert_allclose(result_avg, expected_avg)
    np.testing.assert_allclose(result_dense, expected_dense)


def test_rank_empty_array_returns_empty():
    """Test empty input handling."""
    result = rank(np.array([]), method="average")
    
    assert result.size == 0
    assert result.dtype == np.float64


def test_rank_invalid_method_raises():
    """Test invalid method parameter."""
    data = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="method must be one of"):
        rank(data, method="invalid")


def test_rank_non_1d_input_raises():
    """Test 2D input handling."""
    data = np.array([[1, 2], [3, 4]])
    
    with pytest.raises(ValueError, match="1D array"):
        rank(data)


def test_rank_default_method_is_average():
    """Test default method behavior."""
    data = np.array([10, 20, 20, 30])
    
    result = rank(data)
    
    expected = np.array([1.0, 2.5, 2.5, 4.0])
    np.testing.assert_allclose(result, expected)


def test_percentile_basic():
    """Test basic percentile computation."""
    data = np.array([1, 2, 3, 4, 5])
    
    p25 = percentile(data, 25)
    p50 = percentile(data, 50)
    p75 = percentile(data, 75)
    
    assert np.isclose(p25, 2.0)
    assert np.isclose(p50, 3.0)
    assert np.isclose(p75, 4.0)


def test_percentile_multiple_quantiles():
    """Test multiple quantiles at once."""
    data = np.array([1, 2, 3, 4, 5])
    
    result = percentile(data, [25, 50, 75])
    
    expected = np.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(result, expected)


def test_percentile_different_interpolations():
    """Test different interpolation methods."""
    data = np.array([1, 2, 3])
    
    p50_linear = percentile(data, 50, interpolation="linear")
    p50_lower = percentile(data, 50, interpolation="lower")
    p50_higher = percentile(data, 50, interpolation="higher")
    p50_midpoint = percentile(data, 50, interpolation="midpoint")
    
    assert np.isclose(p50_linear, 2.0)
    assert p50_lower == 2.0
    assert p50_higher == 2.0
    assert np.isclose(p50_midpoint, 2.0)


def test_percentile_nan_ignored():
    """Test NaN handling in percentiles."""
    data = np.array([1, np.nan, 3, np.nan, 5])
    
    p50 = percentile(data, 50)
    
    # Median of [1, 3, 5] = 3
    assert np.isclose(p50, 3.0)


def test_percentile_q_out_of_bounds_raises():
    """Test invalid q values."""
    data = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="within the range"):
        percentile(data, 101)
    
    with pytest.raises(ValueError, match="within the range"):
        percentile(data, -1)


def test_percentile_invalid_interpolation_raises():
    """Test invalid interpolation method."""
    data = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="interpolation must be one of"):
        percentile(data, 50, interpolation="invalid")


def test_percentile_empty_array_raises():
    """Test empty array handling."""
    with pytest.raises(ValueError, match="must not be empty"):
        percentile(np.array([]), 50)


def test_rank_stability_with_identical_order():
    """Test that stable sorting preserves order for ties."""
    data = np.array([3, 1, 1, 2])
    indices = np.array([0, 1, 2, 3])  # Original order tracking
    
    ranked = rank(data, method="ordinal")
    expected_ranks = np.array([3.0, 1.0, 2.0, 4.0])  # Positions: 3rd=1st, 1st=1st, 2nd=1st, 4th=2nd
    
    np.testing.assert_allclose(ranked, expected_ranks)