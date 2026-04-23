"""
Tests for io.py module
"""

import numpy as np
from numcompute.io import load_csv


def test_load_simple_csv(tmp_path):
    csv_file = tmp_path / "simple.csv"
    csv_file.write_text(
        "age,grade\n"
        "20,85\n"
        "22,90\n"
        "21,88\n"
    )

    data = load_csv(str(csv_file))

    assert data.shape == (3, 2)
    assert data[0, 0] == 20.0
    assert data[0, 1] == 85.0


def test_load_csv_with_missing_values(tmp_path):
    csv_file = tmp_path / "missing.csv"
    csv_file.write_text(
        "age,grade\n"
        "20,85\n"
        "22,\n"
        "21,88\n"
    )

    data = load_csv(str(csv_file))

    assert data.shape == (3, 2)
    assert np.isnan(data[1, 1])


def test_load_csv_custom_delimiter(tmp_path):
    csv_file = tmp_path / "tabs.csv"
    csv_file.write_text(
        "age\tgrade\n"
        "20\t85\n"
        "22\t90\n"
    )

    data = load_csv(str(csv_file), delimiter='\t')

    assert data.shape == (2, 2)
    assert data[0, 0] == 20.0