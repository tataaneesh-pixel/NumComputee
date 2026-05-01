import numpy as np
import pytest

from numcompute.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)


def test_standard_scaler_basic_mean_and_std():
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert X_scaled.shape == (3, 2)
    assert np.allclose(np.mean(X_scaled, axis=0), [0.0, 0.0], atol=1e-10)
    assert np.allclose(np.std(X_scaled, axis=0), [1.0, 1.0], atol=1e-10)


def test_standard_scaler_constant_feature_becomes_zero():
    X = np.array([[5.0, 2.0],
                  [5.0, 4.0],
                  [5.0, 6.0]])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert not np.any(np.isnan(X_scaled))
    assert np.allclose(X_scaled[:, 0], 0.0)


def test_standard_scaler_transform_before_fit_raises():
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0]])

    scaler = StandardScaler()

    with pytest.raises(RuntimeError):
        scaler.transform(X)


def test_standard_scaler_feature_mismatch_raises():
    X_train = np.array([[1.0, 2.0],
                        [3.0, 4.0]])
    X_test = np.array([[1.0, 2.0, 3.0]])

    scaler = StandardScaler().fit(X_train)

    with pytest.raises(ValueError):
        scaler.transform(X_test)


def test_standard_scaler_1d_input_returns_2d():
    X = np.array([1.0, 2.0, 3.0, 4.0])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert X_scaled.ndim == 2
    assert X_scaled.shape == (4, 1)
    assert np.allclose(np.mean(X_scaled, axis=0), [0.0], atol=1e-10)


def test_minmax_scaler_basic_range_zero_one():
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    assert X_scaled.shape == (3, 2)
    assert np.allclose(np.min(X_scaled, axis=0), [0.0, 0.0])
    assert np.allclose(np.max(X_scaled, axis=0), [1.0, 1.0])


def test_minmax_scaler_custom_range():
    X = np.array([[1.0],
                  [2.0],
                  [3.0],
                  [4.0],
                  [5.0]])

    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(np.min(X_scaled), -1.0)
    assert np.allclose(np.max(X_scaled), 1.0)


def test_minmax_scaler_constant_feature_maps_to_low_end():
    X = np.array([[7.0],
                  [7.0],
                  [7.0]])

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(X_scaled, np.zeros((3, 1)))


def test_minmax_scaler_invalid_feature_range_raises():
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1.0, 1.0))


def test_onehot_encoder_single_feature():
    X = np.array([[0],
                  [1],
                  [0],
                  [2]])

    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)

    expected = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    assert X_encoded.shape == (4, 3)
    np.testing.assert_array_equal(X_encoded, expected)


def test_onehot_encoder_multiple_features():
    X = np.array([
        [0, 1],
        [1, 0],
        [0, 0],
    ])

    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)

    assert X_encoded.shape == (3, 4)


def test_onehot_encoder_transform_before_fit_raises():
    X = np.array([[0], [1]])

    encoder = OneHotEncoder()

    with pytest.raises(RuntimeError):
        encoder.transform(X)


def test_onehot_encoder_feature_mismatch_raises():
    X_train = np.array([[0, 1],
                        [1, 0]])
    X_test = np.array([[0, 1, 2]])

    encoder = OneHotEncoder().fit(X_train)

    with pytest.raises(ValueError):
        encoder.transform(X_test)


def test_onehot_encoder_1d_input_supported():
    X = np.array(["red", "blue", "red"], dtype=object)

    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)

    assert X_encoded.ndim == 2
    assert X_encoded.shape == (3, 2)


def test_simple_imputer_mean_strategy():
    X = np.array([
        [1.0, np.nan],
        [3.0, 6.0],
        [5.0, 8.0],
    ])

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    expected = np.array([
        [1.0, 7.0],
        [3.0, 6.0],
        [5.0, 8.0],
    ])

    np.testing.assert_allclose(X_imputed, expected)


def test_simple_imputer_constant_strategy():
    X = np.array([
        [1.0, np.nan],
        [np.nan, 4.0],
    ])

    imputer = SimpleImputer(strategy="constant", fill_value=-1.0)
    X_imputed = imputer.fit_transform(X)

    expected = np.array([
        [1.0, -1.0],
        [-1.0, 4.0],
    ])

    np.testing.assert_allclose(X_imputed, expected)


def test_simple_imputer_transform_before_fit_raises():
    X = np.array([[1.0, np.nan]])

    imputer = SimpleImputer(strategy="mean")

    with pytest.raises(RuntimeError):
        imputer.transform(X)


def test_simple_imputer_invalid_strategy_raises():
    with pytest.raises(ValueError):
        SimpleImputer(strategy="median")


def test_simple_imputer_feature_mismatch_raises():
    X_train = np.array([
        [1.0, np.nan],
        [2.0, 3.0],
    ])
    X_test = np.array([[1.0, 2.0, 3.0]])

    imputer = SimpleImputer(strategy="mean").fit(X_train)

    with pytest.raises(ValueError):
        imputer.transform(X_test)