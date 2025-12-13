# build_model_to_predict_washroom.py
# author: William Song
# date: 2025-12-06

import click
import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score, ConfusionMatrixDisplay
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.evaluate_workflow import (
    evaluate_pipeline,
    save_evaluation_csv
)


def evaluate_pipeline(pipeline, X_test, y_test):
    """
    Evaluate a fitted pipeline on test data.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline object.
    X_test : pandas.DataFrame
        Test features.
    y_test : pandas.Series
        Test labels.

    Returns
    -------
    tuple
        accuracy : float
            Classification accuracy on test data.
        f2_score : float
            F2 score (beta=2) with positive label 'Y'.
        y_pred : numpy.ndarray
            Predicted labels for test data.
        cm_table : pandas.DataFrame
            Confusion matrix table (counts of true vs predicted).
    """
    accuracy = pipeline.score(X_test, y_test)
    y_pred = pipeline.predict(X_test)
    f2_score = fbeta_score(
        y_test, 
        y_pred, 
        beta=2, 
        pos_label="Y"
    )
    cm_table = pd.crosstab(y_test, y_pred)
    return accuracy, f2_score, y_pred, cm_table


def save_evaluation_csv(accuracy, f2_score, y_test, y_pred, cm_table, results_to):
    """
    Save evaluation results (scores, predictions, confusion matrix) as CSV files.

    Parameters
    ----------
    accuracy : float
        Classification accuracy.
    f2_score : float
        F2 score (beta=2).
    y_test : pandas.Series
        True labels.
    y_pred : numpy.ndarray
        Predicted labels.
    cm_table : pandas.DataFrame
        Confusion matrix table.
    results_to : str
        Directory path to save CSV outputs.
    """
    pd.DataFrame({
        "accuracy": [accuracy],
        "F2 score (beta = 2)": [f2_score]
    }).to_csv(os.path.join(results_to, "test_scores.csv"), index=False)

    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
        os.path.join(results_to, "test_predictions.csv"), index=False
    )

    cm_table.to_csv(os.path.join(results_to, "svm_confusion_matrix.csv"))


@click.command()
@click.option('--test-data', type=str, help="Path to test data")
@click.option('--pipeline-from', type=str, help="Path to directory where the fit pipeline object lives")
@click.option('--results-to', type=str, help="Path to directory where cross validation table will be written to")
@click.option('--viz-to', type=str, help="Path to directory where visualizations will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)

def main(test_data, pipeline_from, results_to, viz_to, seed):
    '''Evaluates the parks washroom classifier on the test data 
    and saves the evaluation results.'''
    np.random.seed(seed)

    # read in data & split
    target = "Washrooms"
    test_df = pd.read_csv(test_data)
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # read in pipeline object
    with open(pipeline_from, 'rb') as f:
        parks_fit = pickle.load(f)

    # Evaluate
    accuracy, f2_score, y_pred, cm_table = evaluate_pipeline(parks_fit, X_test, y_test)

    # Save CSV outputs
    save_evaluation_csv(accuracy, f2_score, y_test, y_pred, cm_table, results_to)

    # Create and save contingency table into plot
    cm = ConfusionMatrixDisplay.from_estimator(
        parks_fit,
        X_test,
        y_test,
        values_format="d"
    )
    cm.plot()    
    plt.savefig(os.path.join(viz_to, "svm_confusion_matrix.png"), dpi=200)
    plt.close()


if __name__ == '__main__':
    main()