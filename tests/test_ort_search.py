import numpy as np
import pytest

from numcompute.sort_search import (
    binary_search,
    multi_key_sort,
    quickselect,
    stable_sort,
    topk,
)


def test_stable_sort_basic_1d():
    arr = np.array([3, 1, 2, 1])
    result = stable_sort(arr)

    expected = np.array([1, 1, 2, 3])
    np.testing.assert_array_equal(result, expected)


def test_stable_sort_2d_axis_0():
    arr = np.array([
        [3, 2],
        [1, 4],
        [2, 1],
    ])
    result = stable_sort(arr, axis=0)

    expected = np.array([
        [1, 1],
        [2, 2],
        [3, 4],
    ])
    np.testing.assert_array_equal(result, expected)


def test_stable_sort_empty_raises():
    with pytest.raises(ValueError):
        stable_sort(np.array([]))


def test_multi_key_sort_primary_then_secondary():
    arr = np.array([
        [2, 3],
        [1, 2],
        [2, 1],
        [1, 1],
    ])
    result = multi_key_sort(arr, keys=[0, 1], ascending=True)

    expected = np.array([
        [1, 1],
        [1, 2],
        [2, 1],
        [2, 3],
    ])
    np.testing.assert_array_equal(result, expected)


def test_multi_key_sort_descending():
    arr = np.array([
        [2, 3],
        [1, 2],
        [2, 1],
        [1, 1],
    ])
    result = multi_key_sort(arr, keys=[0, 1], ascending=False)

    expected = np.array([
        [2, 3],
        [2, 1],
        [1, 2],
        [1, 1],
    ])
    np.testing.assert_array_equal(result, expected)


def test_multi_key_sort_invalid_dimension_raises():
    arr = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        multi_key_sort(arr, keys=[0])


def test_multi_key_sort_empty_keys_raises():
    arr = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        multi_key_sort(arr, keys=[])


def test_multi_key_sort_out_of_bounds_key_raises():
    arr = np.array([[1, 2], [3, 4]])

    with pytest.raises(IndexError):
        multi_key_sort(arr, keys=[2])


def test_topk_largest_returns_sorted_values_and_indices():
    values = np.array([10, 30, 20, 40, 5])
    top_values, top_indices = topk(values, k=3, largest=True, return_indices=True)

    np.testing.assert_array_equal(top_values, np.array([40, 30, 20]))
    np.testing.assert_array_equal(top_values, values[top_indices])


def test_topk_smallest_returns_sorted_values_and_indices():
    values = np.array([10, 30, 20, 40, 5])
    top_values, top_indices = topk(values, k=2, largest=False, return_indices=True)

    np.testing.assert_array_equal(top_values, np.array([5, 10]))
    np.testing.assert_array_equal(top_values, values[top_indices])


def test_topk_without_indices():
    values = np.array([4, 1, 9, 2])
    top_values = topk(values, k=2, largest=True, return_indices=False)

    np.testing.assert_array_equal(top_values, np.array([9, 4]))


def test_topk_k_equals_length():
    values = np.array([3, 1, 2])
    top_values, top_indices = topk(values, k=3, largest=True, return_indices=True)

    np.testing.assert_array_equal(top_values, np.array([3, 2, 1]))
    np.testing.assert_array_equal(top_values, values[top_indices])


def test_topk_invalid_k_zero_raises():
    values = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        topk(values, k=0)


def test_topk_invalid_k_too_large_raises():
    values = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        topk(values, k=4)


def test_topk_non_integer_k_raises():
    values = np.array([1, 2, 3])

    with pytest.raises(TypeError):
        topk(values, k=2.5)


def test_topk_non_1d_input_raises():
    values = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        topk(values, k=2)


def test_quickselect_returns_kth_smallest():
    arr = np.array([7, 2, 9, 1, 5])
    result = quickselect(arr, 2)

    assert result == 5


def test_quickselect_with_duplicates():
    arr = np.array([4, 2, 2, 9, 7])
    result = quickselect(arr, 1)

    assert result == 2


def test_quickselect_first_and_last_order_statistics():
    arr = np.array([8, 3, 6, 1])

    assert quickselect(arr, 0) == 1
    assert quickselect(arr, 3) == 8


def test_quickselect_empty_raises():
    with pytest.raises(ValueError):
        quickselect(np.array([]), 0)


def test_quickselect_invalid_k_raises():
    arr = np.array([1, 2, 3])

    with pytest.raises(IndexError):
        quickselect(arr, 3)


def test_binary_search_existing_value():
    arr = np.array([1, 3, 5, 7, 9])

    idx, exists = binary_search(arr, 5)

    assert idx == 2
    assert exists is True


def test_binary_search_missing_value_insertion_point():
    arr = np.array([1, 3, 5, 7, 9])

    idx, exists = binary_search(arr, 6)

    assert idx == 3
    assert exists is False


def test_binary_search_leftmost_duplicate():
    arr = np.array([1, 2, 2, 2, 3])

    idx, exists = binary_search(arr, 2)

    assert idx == 1
    assert exists is True


def test_binary_search_non_1d_input_raises():
    arr = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        binary_search(arr, 2)