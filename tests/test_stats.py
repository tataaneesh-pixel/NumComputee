import numpy as np
import pytest

from numcompute.stats import descriptive_stats, histogram, quantile


def test_descriptive_stats_basic_no_nan():
    """Test descriptive stats on clean numeric data."""
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])

    stats = descriptive_stats(X)

    np.testing.assert_allclose(stats["mean"], [3.0, 4.0])
    np.testing.assert_allclose(stats["median"], [3.0, 4.0])
    np.testing.assert_allclose(stats["std"], [2.0, 2.0], atol=1e-10)
    np.testing.assert_allclose(stats["min"], [1.0, 2.0])
    np.testing.assert_allclose(stats["max"], [5.0, 6.0])


def test_descriptive_stats_axis_0():
    """Test axis=0 behavior."""
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])

    stats = descriptive_stats(X, axis=0)

    assert stats["mean"].shape == (2,)
    np.testing.assert_allclose(stats["mean"], [3.0, 4.0])


def test_descriptive_stats_axis_1():
    """Test axis=1 behavior."""
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])

    stats = descriptive_stats(X, axis=1)

    assert stats["mean"].shape == (3,)
    np.testing.assert_allclose(stats["mean"], [1.5, 3.5, 5.5])


def test_descriptive_stats_nan_ignored():
    """Test NaN handling in descriptive stats."""
    X = np.array([[1.0, np.nan],
                  [3.0, 4.0],
                  [np.nan, 6.0]])

    stats = descriptive_stats(X)

    np.testing.assert_allclose(stats["mean"], [2.0, 5.0])


def test_descriptive_stats_empty_raises():
    """Test empty array handling."""
    with pytest.raises(ValueError):
        descriptive_stats(np.array([]))


def test_quantile_basic():
    """Test basic quantile computation."""
    X = np.array([1, 2, 3, 4, 5])

    q25 = quantile(X, 0.25)
    q50 = quantile(X, 0.5)
    q75 = quantile(X, 0.75)

    assert np.isclose(q25, 2.0)
    assert np.isclose(q50, 3.0)
    assert np.isclose(q75, 4.0)


def test_quantile_multiple_quantiles():
    """Test multiple quantiles at once."""
    X = np.array([1, 2, 3, 4, 5])

    result = quantile(X, [0.25, 0.5, 0.75])

    expected = np.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(result, expected)


def test_quantile_axis_behavior():
    """Test axis behavior in quantiles."""
    X = np.array([[1, 4],
                  [2, 5],
                  [3, 6]])

    q50_axis0 = quantile(X, 0.5, axis=0)
    q50_axis1 = quantile(X, 0.5, axis=1)

    np.testing.assert_allclose(q50_axis0, [2.0, 5.0])
    np.testing.assert_allclose(q50_axis1, [2.0, 5.0])


def test_quantile_nan_ignored():
    """Test NaN handling in quantiles."""
    X = np.array([1, np.nan, 3, np.nan, 5])

    q50 = quantile(X, 0.5)

    assert np.isclose(q50, 3.0)


def test_quantile_q_out_of_bounds_raises():
    """Test invalid q values."""
    X = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        quantile(X, 1.1)

    with pytest.raises(ValueError):
        quantile(X, -0.1)


def test_quantile_empty_raises():
    """Test empty array handling."""
    with pytest.raises(ValueError):
        quantile(np.array([]), 0.5)


def test_histogram_basic():
    """Test basic histogram computation."""
    X = np.array([1, 2, 2, 3, 4, 5, 5, 5])

    counts, edges = histogram(X, bins=3)

    assert counts.shape == (3,)
    np.testing.assert_allclose(counts, [1, 2, 5])
    assert edges.shape == (4,)
    assert np.isclose(edges[0], 1.0)
    assert np.isclose(edges[-1], 5.0)


def test_histogram_custom_bins():
    """Test custom bin specification."""
    X = np.array([1, 2, 3, 4])

    counts, edges = histogram(X, bins=[1, 2, 3, 4])

    np.testing.assert_array_equal(counts, [1, 1, 1])
    np.testing.assert_array_equal(edges, [1, 2, 3, 4])


def test_histogram_range_parameter():
    """Test range parameter."""
    X = np.array([1, 3, 5, 7])

    counts, edges = histogram(X, bins=2, range=(0, 6))

    np.testing.assert_array_equal(counts, [2, 1])
    np.testing.assert_allclose(edges, [0, 3, 6])


def test_histogram_all_nan_raises():
    """Test all-NaN input handling."""
    X = np.array([np.nan, np.nan, np.nan])

    with pytest.raises(ValueError, match="all values are NaN"):
        histogram(X)


def test_histogram_empty_raises():
    """Test empty array handling."""
    with pytest.raises(ValueError):
        histogram(np.array([]))


def test_histogram_2d_flattened():
    """Test 2D input is flattened."""
    X = np.array([[1, 2],
                  [3, 4]])

    counts, edges = histogram(X, bins=2)

    assert counts.shape == (2,)
    assert np.sum(counts) == 4