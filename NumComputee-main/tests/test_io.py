import numpy as np
import pytest

from numcompute.io import load_csv


def test_load_csv_basic_numeric_with_header(tmp_path):
    file_path = tmp_path / "students.csv"
    file_path.write_text("a,b,c\n1,2,3\n4,5,6\n")

    data = load_csv(file_path)

    expected = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]])
    assert data.shape == (2, 3)
    np.testing.assert_allclose(data, expected)


def test_load_csv_without_header(tmp_path):
    file_path = tmp_path / "numbers.csv"
    file_path.write_text("1,2,3\n4,5,6\n")

    data = load_csv(file_path, skip_header=False)

    expected = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]])
    assert data.shape == (2, 3)
    np.testing.assert_allclose(data, expected)


def test_load_csv_with_missing_values_as_nan(tmp_path):
    file_path = tmp_path / "missing.csv"
    file_path.write_text("x,y,z\n1,,3\n4,5,\n")

    data = load_csv(file_path)

    assert data.shape == (2, 3)
    assert np.isnan(data[0, 1])
    assert np.isnan(data[1, 2])
    assert data[0, 0] == 1.0
    assert data[1, 1] == 5.0


def test_load_csv_with_custom_fill_value(tmp_path):
    file_path = tmp_path / "filled.csv"
    file_path.write_text("x,y,z\n1,,3\n4,5,\n")

    data = load_csv(file_path, filling_values=0.0)

    expected = np.array([[1.0, 0.0, 3.0],
                         [4.0, 5.0, 0.0]])
    np.testing.assert_allclose(data, expected)


def test_load_csv_tab_delimiter(tmp_path):
    file_path = tmp_path / "tabbed.tsv"
    file_path.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n")

    data = load_csv(file_path, delimiter="\t")

    expected = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(data, expected)


def test_load_csv_single_data_row_returns_2d(tmp_path):
    file_path = tmp_path / "single_row.csv"
    file_path.write_text("a,b,c\n7,8,9\n")

    data = load_csv(file_path)

    expected = np.array([[7.0, 8.0, 9.0]])
    assert data.shape == (1, 3)
    np.testing.assert_allclose(data, expected)


def test_load_csv_single_scalar_returns_2d(tmp_path):
    file_path = tmp_path / "single_value.csv"
    file_path.write_text("a\n42\n")

    data = load_csv(file_path)

    expected = np.array([[42.0]])
    assert data.shape == (1, 1)
    np.testing.assert_allclose(data, expected)


def test_load_csv_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        load_csv("this_file_does_not_exist.csv")


def test_load_csv_empty_file_raises_value_error(tmp_path):
    file_path = tmp_path / "empty.csv"
    file_path.write_text("")

    with pytest.raises(ValueError, match="empty|readable data|Empty|empty"):
        load_csv(file_path, skip_header=False)


def test_load_csv_dtype_int(tmp_path):
    file_path = tmp_path / "ints.csv"
    file_path.write_text("a,b\n1,2\n3,4\n")

    data = load_csv(file_path, dtype=np.int64)

    expected = np.array([[1, 2],
                         [3, 4]], dtype=np.int64)
    assert data.dtype == np.int64
    np.testing.assert_array_equal(data, expected)