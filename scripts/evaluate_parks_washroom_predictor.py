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
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay

@click.command()
@click.option('--scaled-train-data', type=str, help="Path to scaled train data")
@click.option('--scaled-test-data', type=str, help="Path to scaled test data")
@click.option('--y-data', type=str, help="Path to y data")
@click.option('--results-to', type=str, help="Path to directory where cross validation table will be written to")
@click.option('--pipeline-to', type=str, help="Path to directory where the model object will be written to")
@click.option('--viz-to', type=str, help="Path to directory where visualizations will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)

def main(train_data, preprocessor, results_to, pipeline_to, viz_to, seed):
    '''Fits the parks washroom classifier on the training data 
    and saves the pipeline object results.'''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    # read in data & cancer_fit (pipeline object)
    cancer_test = pd.read_csv(scaled_test_data)
    if columns_to_drop:
        to_drop = pd.read_csv(columns_to_drop).feats_to_drop.tolist()
        cancer_test = cancer_test.drop(columns=to_drop)
    with open(pipeline_from, 'rb') as f:
        parks_fit = pickle.load(f)


    # Compute accuracy
    accuracy = cancer_fit.score(
        cancer_test.drop(columns=["class"]),
        cancer_test["class"]
    )

    # Compute F2 score (beta = 2)
    cancer_preds = cancer_test.assign(
        predicted=cancer_fit.predict(cancer_test)
    )
    f2_beta_2_score = fbeta_score(
        cancer_preds['class'],
        cancer_preds['predicted'],
        beta=2,
        pos_label='Malignant'
    )


    test_scores = pd.DataFrame({'accuracy': [accuracy], 'F2 score (beta = 2)': [f2_beta_2_score]})
    test_scores.to_csv(os.path.join(results_to, "test_scores.csv"), index=False)

    confusion_matrix = pd.crosstab(
        cancer_preds["class"],
        cancer_preds["predicted"],
    )
    confusion_matrix.to_csv(os.path.join(results_to, "confusion_matrix.csv"))


    

    # Create Contingency Table Plot
    cm = ConfusionMatrixDisplay.from_estimator(
        pipe_svm_rbf_fit,
        X_train,
        y_train,
        #X_test,
        #y_test,
        values_format="d"
    )
    cm.plot()
    
    plt.savefig(os.path.join(viz_to, "svm_confusion_matrix.png"), dpi=200)
    plt.close()


if __name__ == '__main__':
    main()