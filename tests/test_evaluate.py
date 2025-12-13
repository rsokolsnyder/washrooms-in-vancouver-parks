import os
import sys
import pickle
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")  # avoid GUI backend issues in tests

from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline

# Ensure project root is on path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.evaluate_parks_washroom_predictor import (
    evaluate_pipeline,
    save_evaluation_csv
)


# --- Fixtures ---
@pytest.fixture
def sample_data():
    """Synthetic dataset for expected use cases."""
    X = pd.DataFrame(np.random.randn(20, 3), columns=["f1", "f2", "f3"])
    y = pd.Series(np.random.choice(["Y", "N"], size=20))
    return X, y


@pytest.fixture
def tiny_data():
    """Edge case: very small dataset (only 2 samples)."""
    X = pd.DataFrame([[0, 1], [1, 0]], columns=["f1", "f2"])
    y = pd.Series(["Y", "N"])
    return X, y


@pytest.fixture
def pipeline():
    """Simple pipeline for testing."""
    return make_pipeline(StandardScaler(), DummyClassifier(strategy="most_frequent"))


# --- Expected use cases ---
def test_evaluate_pipeline_expected(sample_data, pipeline):
    X, y = sample_data
    pipeline.fit(X, y)
    accuracy, f2_score, y_pred, cm_table = evaluate_pipeline(pipeline, X, y)
    assert isinstance(accuracy, float)
    assert isinstance(f2_score, float)
    assert len(y_pred) == len(y)
    assert isinstance(cm_table, pd.DataFrame)


def test_save_evaluation_csv_expected(tmp_path, sample_data, pipeline):
    X, y = sample_data
    pipeline.fit(X, y)
    accuracy, f2_score, y_pred, cm_table = evaluate_pipeline(pipeline, X, y)

    results_to = tmp_path
    save_evaluation_csv(accuracy, f2_score, y, y_pred, cm_table, str(results_to))

    assert (results_to / "test_scores.csv").exists()
    assert (results_to / "test_predictions.csv").exists()
    assert (results_to / "svm_confusion_matrix.csv").exists()


# --- Edge cases ---
def test_evaluate_pipeline_tiny_data(tiny_data, pipeline):
    X, y = tiny_data
    pipeline.fit(X, y)
    accuracy, f2_score, y_pred, cm_table = evaluate_pipeline(pipeline, X, y)
    assert len(y_pred) == len(y)


# --- Error cases ---
def test_evaluate_pipeline_invalid_model(sample_data):
    X, y = sample_data
    # Passing a non-pipeline should raise AttributeError
    with pytest.raises(AttributeError):
        evaluate_pipeline("not_a_pipeline", X, y)


def test_save_evaluation_csv_invalid_dir(sample_data, pipeline):
    X, y = sample_data
    pipeline.fit(X, y)
    accuracy, f2_score, y_pred, cm_table = evaluate_pipeline(pipeline, X, y)

    # Expect OSError when directory doesn't exist
    with pytest.raises(OSError):
        save_evaluation_csv(accuracy, f2_score, y, y_pred, cm_table, "nonexistent_dir")
