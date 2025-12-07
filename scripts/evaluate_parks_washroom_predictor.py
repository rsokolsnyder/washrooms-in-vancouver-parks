# build_model_to_predict_washroom.py
# author: William Song
# date: 2025-12-06

import click
import os
import numpy as np
import pandas as pd
import pickle
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import ConfusionMatrixDisplay

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
    set_config(transform_output="default")

    # read in data & split
    target = "Washrooms"
    test_df = pd.read_csv(test_data)
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # read in pipeline object
    with open(pipeline_from, 'rb') as f:
        parks_fit = pickle.load(f)

    # Compute accuracy
    accuracy = parks_fit.score(X_test, y_test)

    # Predictions
    y_pred = parks_fit.predict(X_test)
    
    # Compute F2 score (beta = 2)
    f2_beta_2_score = fbeta_score(
        y_test,
        y_pred,
        beta=2,
        pos_label='Y'
    )

    # Save scores
    test_scores = pd.DataFrame({
        'accuracy': [accuracy], 
        'F2 score (beta = 2)': [f2_beta_2_score]
    })
    test_scores.to_csv(os.path.join(results_to, "test_scores.csv"), index=False)

    # Save predictions
    preds_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    })
    preds_df.to_csv(os.path.join(results_to, "test_predictions.csv"), index=False)

    # Create and save contingency table in both text and plot
    confusion_matrix = pd.crosstab(
        y_test,
        y_pred,
    )
    confusion_matrix.to_csv(os.path.join(results_to, "svm_confusion_matrix.csv"))
    
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