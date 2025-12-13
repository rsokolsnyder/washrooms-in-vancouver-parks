import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import tempfile
import pickle
import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scripts.fit_parks_washroom_classifier import (
    mean_std_cross_val_scores,
    build_pipeline,
    evaluate_and_save,
    merge_results,
    fit_and_save
)


@pytest.fixture
def sample_data():
    """Create a small synthetic dataset for testing."""
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
def preprocessor():
    """Simple preprocessor for testing."""
    return StandardScaler()


def test_mean_std_cross_val_scores(sample_data, preprocessor):
    X, y = sample_data
    pipe = build_pipeline(preprocessor, DummyClassifier(strategy="most_frequent"))
    scores = mean_std_cross_val_scores(pipe, X, y, cv=3)
    assert isinstance(scores, pd.Series)
    assert "test_score" in scores.index


def test_build_pipeline(preprocessor):
    model = DummyClassifier(strategy="most_frequent")
    pipe = build_pipeline(preprocessor, model)
    # Pipeline should have two steps
    assert len(pipe.steps) == 2


def test_evaluate_and_save(tmp_path, sample_data, preprocessor):
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


def test_merge_results(tmp_path):
    df1 = pd.DataFrame({"dummy": ["0.5 (+/- 0.1)"]})
    df2 = pd.DataFrame({"svc": ["0.9 (+/- 0.05)"]})
    results_to = tmp_path
    merged = merge_results([df1, df2], str(results_to))
    assert "dummy" in merged.columns
    assert "svc" in merged.columns
    assert (results_to / "combined_result.csv").exists()


def test_fit_and_save(tmp_path, sample_data, preprocessor):
    X, y = sample_data
    pipe = build_pipeline(preprocessor, DummyClassifier(strategy="most_frequent"))
    pipeline_to = tmp_path
    fitted = fit_and_save(pipe, X, y, str(pipeline_to), "dummy")
    assert hasattr(fitted, "predict")
    # Check pickle exists
    assert (pipeline_to / "pipe_dummy_fully_trained.pickle").exists()
    # Load pickle and test predict
    with open(pipeline_to / "pipe_dummy_fully_trained.pickle", "rb") as f:
        loaded = pickle.load(f)
    preds = loaded.predict(X)
    assert len(preds) == len(y)