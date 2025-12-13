# build_model_to_predict_washroom.py
# author: William Song
# date: 2025-12-06

import click
import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import functions from src
from src.fit_workflow import (
    mean_std_cross_val_scores,
    evaluate_and_save,
    merge_results,
)


@click.command()
@click.option(
    "--train-data", 
    type=str, 
    help="Path to train data"
)
@click.option(
    "--preprocessor", 
    type=str, 
    help="Path to parks_preprocessor.pickle"
)
@click.option(
    "--results-to", 
    type=str, 
    help="Path to directory where cross validation table will be written to"
)
@click.option(
    "--pipeline-to", 
    type=str, 
    help="Path to directory where the model object will be written to"
)
@click.option(
    "--seed", 
    type=int, 
    help="Random seed", default=123
)
    
def main(train_data, preprocessor, results_to, pipeline_to, seed):
    '''Fits the parks washroom classifier on the training data 
    and saves the pipeline object results.'''
    np.random.seed(seed)
    target = "Washrooms"
    
    # Load data and preprocessor
    train_df = pd.read_csv(train_data)
    data_preprocessor = pickle.load(open(preprocessor, "rb"))
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    
    # Define models
    models = {
        "dummy": DummyClassifier(random_state=seed),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm_rbf": SVC(kernel="rbf", C=1.0, gamma="scale"),
    }
    
    # Evaluate all models
    results_list = []
    pipes = {}
    for name, model in models.items():
        df, pipe = evaluate_and_save(name, model, data_preprocessor, X_train, y_train, results_to, pipeline_to)
        results_list.append(df)
        pipes[name] = pipe
    
    # Merge results
    merge_results(results_list, results_to)
    
    # Fit and save SVM RBF model
    pipe_svm_rbf_fit = pipes["svm_rbf"].fit(X_train, y_train)
    with open(os.path.join(pipeline_to, "pipe_svm_rbf_fully_trained.pickle"), 'wb') as f:
        pickle.dump(pipe_svm_rbf_fit, f)

if __name__ == '__main__':
    main()