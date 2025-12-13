import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import tempfile
import pickle
import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from src.fit_workflow import (
    mean_std_cross_val_scores,
    evaluate_and_save,
    merge_results
)

# Fixtures
@pytest.fixture
def sample_data():
    """Create a small synthetic dataset for testing."""
    X = pd.DataFrame(np.random.randn(20, 3), columns=["f1", "f2", "f3"])
    y = pd.Series(np.random.choice(["Y", "N"], size=20))
    return X, y


@pytest.fixture
def tiny_data():
    """Edge case: very small dataset (only 2 samples per class)."""
    X = pd.DataFrame([[0, 1], [1, 0], [0, 0], [1, 1]], columns=["f1", "f2"])
    y = pd.Series(["Y", "N", "Y", "N"])
    return X, y


@pytest.fixture
def preprocessor():
    """Simple preprocessor for testing."""
    return StandardScaler()

# Expected cases
def test_mean_std_cross_val_scores_expected(sample_data, preprocessor):
    X, y = sample_data
    pipe = make_pipeline(preprocessor, DummyClassifier(strategy="most_frequent"))
    scores = mean_std_cross_val_scores(pipe, X, y, cv=3)
    assert isinstance(scores, pd.Series)
    assert "test_score" in scores.index


def test_evaluate_and_save_expected(tmp_path, sample_data, preprocessor):
    X, y = sample_data
    results_to = tmp_path / "results"
    pipeline_to = tmp_path / "pipes"
    results_to.mkdir()
    pipeline_to.mkdir()

    df, pipe = evaluate_and_save(
        "dummy", DummyClassifier(strategy="most_frequent"),
        preprocessor, X, y, str(results_to), str(pipeline_to), cv=2
    )

    # Check DataFrame
    assert "dummy" in df.columns
    # Check CSV exists
    assert (results_to / "dummy_result.csv").exists()
    # Check pickle exists
    assert (pipeline_to / "pipe_dummy.pickle").exists()


def test_merge_results_expected(tmp_path):
    df1 = pd.DataFrame({"dummy": ["0.5 (+/- 0.1)"]})
    df2 = pd.DataFrame({"svc": ["0.9 (+/- 0.05)"]})
    merged = merge_results([df1, df2], str(tmp_path))
    assert "dummy" in merged.columns
    assert "svc" in merged.columns
    assert (tmp_path / "combined_result.csv").exists()


# Edge cases
def test_mean_std_cross_val_scores_tiny_data(tiny_data, preprocessor):
    X, y = tiny_data
    pipe = make_pipeline(preprocessor, DummyClassifier(strategy="most_frequent"))
    scores = mean_std_cross_val_scores(pipe, X, y, cv=2)
    assert isinstance(scores, pd.Series)


def test_evaluate_and_save_knn_small_data(tmp_path, tiny_data, preprocessor):
    X, y = tiny_data
    results_to = tmp_path / "results"
    pipeline_to = tmp_path / "pipes"
    results_to.mkdir()
    pipeline_to.mkdir()

    df, pipe = evaluate_and_save(
        "knn", DummyClassifier(strategy="most_frequent"),
        preprocessor, X, y, str(results_to), str(pipeline_to), cv=2
    )
    assert "knn" in df.columns


# Error cases
def test_mean_std_cross_val_scores_invalid_model(sample_data):
    X, y = sample_data
    with pytest.raises(Exception):
        mean_std_cross_val_scores("not_a_model", X, y, cv=2)


def test_evaluate_and_save_invalid_paths(sample_data, preprocessor):
    X, y = sample_data
    with pytest.raises(OSError):
        evaluate_and_save(
            "dummy", DummyClassifier(strategy="most_frequent"),
            preprocessor, X, y, "nonexistent_dir", "nonexistent_dir", cv=2
        )


def test_merge_results_empty_list(tmp_path):
    with pytest.raises(IndexError):
        merge_results([], str(tmp_path))