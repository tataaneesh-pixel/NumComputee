import numpy as np
import pytest

from numcompute.pipeline import Pipeline
from numcompute.preprocessing import MinMaxScaler, StandardScaler


def test_pipeline_transformer_chain():
    """Test basic transformer pipeline."""
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])

    pipe = Pipeline([
        ("scale1", StandardScaler()),
        ("scale2", MinMaxScaler()),
    ])

    X_transformed = pipe.fit_transform(X)

    assert X_transformed.shape == (3, 2)
    # Each scaler changes the data range
    assert not np.allclose(X_transformed, X)


def test_pipeline_fit_transform_sequence():
    """Test that fit_transform applies correct sequence."""
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])

    pipe = Pipeline([
        ("minmax", MinMaxScaler()),
        ("standard", StandardScaler()),
    ])

    X1 = pipe.fit_transform(X)
    X2 = StandardScaler().fit_transform(MinMaxScaler().fit_transform(X))

    np.testing.assert_allclose(X1, X2)


def test_pipeline_separate_fit_and_transform():
    """Test fit then transform separately."""
    X_train = np.array([[1.0, 2.0],
                        [3.0, 4.0]])
    X_test = np.array([[5.0, 6.0]])

    pipe = Pipeline([
        ("scale", StandardScaler()),
    ])
    pipe.fit(X_train)
    X_test_transformed = pipe.transform(X_test)

    assert X_test_transformed.shape == (1, 2)


def test_pipeline_step_names_required():
    """Test step names must be strings."""
    with pytest.raises(ValueError, match="non-empty string"):
        Pipeline([("scale", StandardScaler()), (123, StandardScaler())])


def test_pipeline_duplicate_step_names_raises():
    """Test duplicate step names."""
    with pytest.raises(ValueError, match="Duplicate step name"):
        Pipeline([
            ("scale", StandardScaler()),
            ("scale", MinMaxScaler()),
        ])


def test_pipeline_empty_steps_raises():
    """Test empty pipeline."""
    with pytest.raises(ValueError, match="at least one"):
        Pipeline([])


def test_pipeline_feature_mismatch_raises():
    """Test feature dimension mismatch."""
    X_train = np.array([[1.0, 2.0],
                        [3.0, 4.0]])
    X_test = np.array([[1.0]])

    pipe = Pipeline([("scale", StandardScaler())])
    pipe.fit(X_train)

    with pytest.raises(ValueError):
        pipe.transform(X_test)


def test_pipeline_transform_before_fit_raises():
    """Test transform before fit."""
    pipe = Pipeline([("scale", StandardScaler())])
    
    with pytest.raises(RuntimeError):
        pipe.transform(np.array([[1, 2]]))


def test_pipeline_invalid_transformer_raises():
    """Test step without fit/transform methods."""
    class BadStep:
        pass
    
    pipe = Pipeline([("bad", BadStep())])
    
    with pytest.raises(TypeError, match="fit.*transform"):
        pipe.fit_transform(np.array([[1, 2]]))


def test_pipeline_single_step():
    """Test pipeline with single step."""
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0]])

    pipe = Pipeline([("scale", StandardScaler())])
    result = pipe.fit_transform(X)

    assert np.allclose(np.mean(result, axis=0), 0.0, atol=1e-10)


def test_pipeline_access_named_steps():
    """Test named_steps dictionary access."""
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("minmax", MinMaxScaler()),
    ])
    
    assert "scale" in pipe.named_steps
    assert hasattr(pipe.named_steps["scale"], "fit")


def test_pipeline_complex_chain_scaler_composition():
    """Test realistic preprocessing pipeline."""
    X = np.array([[1.0, 100.0],
                  [2.0, 200.0],
                  [3.0, 300.0]])

    pipe = Pipeline([
        ("standard", StandardScaler()),
        ("minmax", MinMaxScaler(feature_range=(-1, 1))),
    ])

    result = pipe.fit_transform(X)

    # After StandardScaler + MinMaxScaler(-1,1), values should be in [-1,1]
    assert np.all(result >= -1.0)
    assert np.all(result <= 1.0)


def test_pipeline_nested_validation():
    """Test that pipeline validates all intermediate steps."""
    class BrokenTransformer:
        def fit(self, X):
            return self
        # Missing transform method intentionally

    pipe = Pipeline([
        ("broken", BrokenTransformer()),
        ("scale", StandardScaler()),
    ])

    with pytest.raises(TypeError, match="transform"):
        pipe.fit_transform(np.array([[1, 2]]))


def test_pipeline_reuse_after_fit():
    """Test pipeline can be reused after fitting."""
    X_train = np.array([[1.0, 2.0],
                        [3.0, 4.0]])
    X_test1 = np.array([[5.0, 6.0]])
    X_test2 = np.array([[7.0, 8.0]])

    pipe = Pipeline([("scale", StandardScaler())])
    pipe.fit(X_train)

    result1 = pipe.transform(X_test1)
    result2 = pipe.transform(X_test2)

    assert result1.shape == (1, 2)
    assert result2.shape == (1, 2)