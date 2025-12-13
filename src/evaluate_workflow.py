import os
import pandas as pd


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