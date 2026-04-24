import numpy as np
import pytest
from numcompute.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

def test_standard_scaler_basic():
    """Test StandardScaler on simple data."""
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6]], dtype=np.float64)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10), "Mean shld be ~0"
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10), "Std shld be ~1"
    
    print("✅ StandardScaler: Mean and std are correct!")
    
def test_standard_scaler_constant_feature():
    X = np.array([[5, 2],
                  [5, 4],
                  [5, 6]], dtype=np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert not np.any(np.isnan(X_scaled)), "Shld not produce NaN"
    assert np.allclose(X_scaled[:, 0], 0), "Constant feature shld scale to 0"
    print("✅ StandardScaler: Handles constant features!")


def test_minmax_scaler_basic():
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6]], dtype=np.float64)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(X_scaled.min(axis=0), 0), "Min shld be 0"
    assert np.allclose(X_scaled.max(axis=0), 1), "Max shld be 1"
    print("✅ MinMaxScaler: Min=0, Max=1 ✓")


def test_minmax_scaler_custom_range():
    X = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(X_scaled.min(), -1), "Min shld be -1"
    assert np.allclose(X_scaled.max(), 1), "Max shld be 1"
    print("✅ MinMaxScaler: Custom range [-1, 1] works!")


def test_onehot_encoder_basic():
    X = np.array([[0],
                  [1],
                  [0],
                  [2]])
    
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)
    assert X_encoded.shape == (4, 3), f"Expected shape (4, 3), got {X_encoded.shape}"
    assert np.array_equal(X_encoded[0], [1, 0, 0]), "First row encoding incorrect"
    assert np.array_equal(X_encoded[1], [0, 1, 0]), "Second row encoding incorrect"
    
    print("✅ OneHotEncoder: Basic encoding works!")


def test_onehot_encoder_multiple_features():
    X = np.array([[0, 0],
                  [1, 1],
                  [0, 1]])
    
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)
    assert X_encoded.shape == (3, 4), f"Expected shape (3, 4), got {X_encoded.shape}"
    print("✅ OneHotEncoder: Multiple features work!")


def test_scaler_transform_before_fit_raises_error():
    X = np.array([[1, 2], [3, 4]])
    scaler = StandardScaler()
    with pytest.raises(RuntimeError):
        scaler.transform(X)
    print("✅ StandardScaler: Raises error when transform called before fit!")