# build_model_to_predict_washroom.py
# author: William Song
# date: 2025-12-02

import click
import os
import numpy as np
import pandas as pd
import pickle
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



# Function for cv score, adapted from 571 lab 2
def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation
    Parameters
    ----------
    model :
    scikit-learn model
    X_train : numpy array or pandas DataFrame
    X in the training data
    y_train :
    y in the training data
    Returns
    ----------
    pandas Series with mean scores from cross_validation
    """
    scores = cross_validate(model, X_train, y_train, **kwargs)
    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []
    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores.iloc[i], std_scores.iloc[i])))
    return pd.Series(data=out_col, index=mean_scores.index)




@click.command()

@click.option('--seed', type=int, help="Random seed", default=123)
def main(scaled_test_data, columns_to_drop, pipeline_from, results_to, seed):
    '''Evaluates the breast cancer classifier on the test data 
    and saves the evaluation results.'''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    # Train Test Split
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # preprocessor for column transformation
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"), OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(), binary_features),
        (categorical_transformer, categorical_features),
        ("drop", drop_features)
    )

    # Setup model and pipline
    model = DummyClassifier(random_state=123)
    pipe = make_pipeline(preprocessor, model)

    dummy_df = pd.DataFrame({
        "dummy" : mean_std_cross_val_scores(pipe, X_train, y_train, cv=5, return_train_score=True)
    })
    dummy_df.transpose()

    # RBF SVC model implementation
    svm_rbf_classifier = SVC(kernel='rbf', C=1.0, gamma='scale') 
    pipe2 = make_pipeline(preprocessor, svm_rbf_classifier)
    svm_rbf_df = pd.DataFrame({
        "svm_rbf" : mean_std_cross_val_scores(pipe2, X_train, y_train, cv=5, return_train_score=True)
    })
    svm_rbf_df.transpose()

    # knn model implementation
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    pipe3 = make_pipeline(preprocessor, knn_classifier)
    knn_df = pd.DataFrame({
        "knn" : mean_std_cross_val_scores(pipe3, X_train, y_train, cv=5, return_train_score=True)
    })
    knn_df.transpose()

    # merge model results together
    result = pd.merge(dummy_df, svm_rbf_df, left_index=True, right_index=True)
    result = pd.merge(result, knn_df, left_index=True, right_index=True)
    result

    # Create Contingency Table
    pipe2.fit(X_train, y_train)

    cm = ConfusionMatrixDisplay.from_estimator(
        pipe2,
        X_test,
        y_test,
        values_format="d"
    )
    cm



if __name__ == '__main__':
    main()