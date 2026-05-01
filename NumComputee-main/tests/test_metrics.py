import numpy as np
import pytest

from numcompute.metrics import (
    accuracy,
    auc,
    confusion_matrix,
    f1,
    mse,
    precision,
    recall,
    roc_curve,
)


def test_accuracy_perfect():
    """Test perfect classification accuracy."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    
    result = accuracy(y_true, y_pred)
    assert np.isclose(result, 1.0)


def test_accuracy_all_wrong():
    """Test zero accuracy."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1])
    
    result = accuracy(y_true, y_pred)
    assert np.isclose(result, 0.0)


def test_accuracy_empty_raises():
    """Test empty input handling."""
    with pytest.raises(ValueError):
        accuracy(np.array([]), np.array([]))


def test_precision_perfect():
    """Test perfect precision."""
    y_true = np.array([0, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 0])
    
    result = precision(y_true, y_pred, pos_label=1)
    assert np.isclose(result, 1.0)


def test_precision_no_true_positives():
    """Test precision when no true positives."""
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])
    
    result = precision(y_true, y_pred, pos_label=1)
    assert np.isclose(result, 0.0)


def test_precision_no_predicted_positives():
    """Test precision when no predicted positives."""
    y_true = np.array([1, 1, 0])
    y_pred = np.array([0, 0, 0])
    
    result = precision(y_true, y_pred, pos_label=1)
    assert np.isclose(result, 0.0)


def test_recall_perfect():
    """Test perfect recall."""
    y_true = np.array([0, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 0])
    
    result = recall(y_true, y_pred, pos_label=1)
    assert np.isclose(result, 1.0)


def test_recall_no_predicted_positives():
    """Test recall when no predicted positives."""
    y_true = np.array([1, 1, 0])
    y_pred = np.array([0, 0, 0])
    
    result = recall(y_true, y_pred, pos_label=1)
    assert np.isclose(result, 0.0)


def test_f1_perfect():
    """Test perfect F1 score."""
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 0])
    
    result = f1(y_true, y_pred, pos_label=1)
    assert np.isclose(result, 1.0)


def test_f1_zero_precision_and_recall():
    """Test F1 when both precision and recall are zero."""
    y_true = np.array([0, 0])
    y_pred = np.array([1, 1])
    
    result = f1(y_true, y_pred, pos_label=1)
    assert np.isclose(result, 0.0)


def test_confusion_matrix_binary():
    """Test binary confusion matrix."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    
    cm = confusion_matrix(y_true, y_pred)
    
    expected = np.array([[2, 0],  # True 0: TN=2, FP=0
                         [1, 1]]) # True 1: FN=1, TP=1
    np.testing.assert_array_equal(cm, expected)


def test_confusion_matrix_multiclass():
    """Test multiclass confusion matrix."""
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 1, 1])
    
    cm = confusion_matrix(y_true, y_pred)
    
    expected = np.array([[1, 1, 0],  # True 0
                         [0, 1, 1],  # True 1
                         [0, 0, 1]]) # True 2
    np.testing.assert_array_equal(cm, expected)


def test_confusion_matrix_custom_labels():
    """Test confusion matrix with custom label ordering."""
    y_true = np.array(["cat", "dog", "cat"])
    y_pred = np.array(["dog", "dog", "cat"])
    
    cm = confusion_matrix(y_true, y_pred, labels=["dog", "cat"])
    
    expected = np.array([[1, 1],  # True dog: TN=1, FP=1
                         [0, 1]])  # True cat: FN=0, TP=1
    np.testing.assert_array_equal(cm, expected)


def test_mse_perfect():
    """Test perfect MSE (zero error)."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    
    result = mse(y_true, y_pred)
    assert np.isclose(result, 0.0)


def test_mse_simple_case():
    """Test MSE on simple differences."""
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    
    result = mse(y_true, y_pred)
    assert np.isclose(result, 0.75)


def test_roc_curve_perfect():
    """Test ROC curve for perfect classifier."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.7, 0.9])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # Perfect classifier: FPR=0, TPR=1 at optimal threshold
    assert np.isclose(fpr[-1], 1.0)
    assert np.isclose(tpr[-1], 1.0)
    assert len(fpr) >= 2


def test_roc_curve_random():
    """Test ROC curve for random classifier."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.6, 0.4, 0.7, 0.3])  # Random ordering
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    # Random classifier should have TPR ≈ FPR throughout
    assert np.allclose(fpr, tpr, atol=0.1)


def test_roc_curve_no_positives_raises():
    """Test ROC curve with no positive samples."""
    y_true = np.array([0, 0, 0])
    y_score = np.array([0.1, 0.5, 0.9])
    
    with pytest.raises(ValueError, match="positive and negative"):
        roc_curve(y_true, y_score)


def test_auc_perfect():
    """Test AUC for perfect classifier."""
    fpr = np.array([0, 0, 1])
    tpr = np.array([0, 1, 1])
    
    result = auc(fpr, tpr)
    assert np.isclose(result, 1.0)


def test_auc_random():
    """Test AUC for random classifier."""
    fpr = np.array([0, 0.5, 1])
    tpr = np.array([0, 0.5, 1])
    
    result = auc(fpr, tpr)
    assert np.isclose(result, 0.5)


def test_auc_single_point_raises():
    """Test AUC with insufficient points."""
    with pytest.raises(ValueError, match="two points"):
        auc(np.array([0]), np.array([1]))


def test_metrics_shape_validation():
    """Test shape validation for all metrics."""
    y_true = np.array([1, 0, 1])
    y_pred_wrong = np.array([0, 1])
    
    with pytest.raises(ValueError, match="same length"):
        accuracy(y_true, y_pred_wrong)


def test_metrics_non_1d_validation():
    """Test non-1D input validation."""
    y_true = np.array([[1], [0], [1]])
    
    with pytest.raises(ValueError, match="1D arrays"):
        accuracy(y_true, y_true)