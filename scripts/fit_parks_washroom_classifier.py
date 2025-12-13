# build_model_to_predict_washroom.py
# author: William Song
# date: 2025-12-06

import click
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# Function for cv score, adapted from 571 lab 2
def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Perform cross-validation and return mean and std scores.

    Parameters
    ----------
    model : scikit-learn estimator or pipeline
        The model to evaluate.
    X_train : pandas DataFrame or numpy array
        Training features.
    y_train : pandas Series or numpy array
        Training labels.
    **kwargs : dict
        Additional arguments passed to sklearn.model_selection.cross_validate.

    Returns
    -------
    pandas.Series
        Series of mean scores with formatted mean and std strings.
    """
    scores = cross_validate(model, X_train, y_train, **kwargs)
    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = [
        f"{mean_scores.iloc[i]:0.3f} (+/- {std_scores.iloc[i]:0.3f})"
        for i in range(len(mean_scores))
    ]
    return pd.Series(data=out_col, index=mean_scores.index)


# Evaluate and save one model
def evaluate_and_save(name, model, preprocessor, X_train, y_train, results_to, pipeline_to, cv=5):
    """
    Evaluate a model with cross-validation, save results and pipeline.

    Parameters
    ----------
    name : str
        Model name identifier.
    model : sklearn estimator
        Model to evaluate.
    preprocessor : sklearn transformer
        Preprocessing object.
    X_train : pandas DataFrame
        Training features.
    y_train : pandas Series
        Training labels.
    results_to : str
        Directory path to save CSV results.
    pipeline_to : str
        Directory path to save pipeline pickle.
    cv : int, optional
        Number of cross-validation folds (default=5).

    Returns
    -------
    tuple
        (results DataFrame, pipeline object).
    """
    pipe = make_pipeline(preprocessor, model)
    scores = mean_std_cross_val_scores(pipe, X_train, y_train, cv=cv, return_train_score=True)
    df = pd.DataFrame({name: scores})
    
    # Save results
    df.transpose().to_csv(os.path.join(results_to, f"{name}_result.csv"), index=True)
    
    # Save pipeline (unfitted)
    with open(os.path.join(pipeline_to, f"pipe_{name}.pickle"), "wb") as f:
        pickle.dump(pipe, f)
    
    return df, pipe


# Merge multiple results
def merge_results(dfs, results_to, filename="combined_result.csv"):
    """
    Merge multiple model results into one CSV.

    Parameters
    ----------
    dfs : list of pandas.DataFrame
        List of result DataFrames to merge.
    results_to : str
        Directory path to save merged CSV.
    filename : str, optional
        Output filename (default="combined_result.csv").

    Returns
    -------
    pandas.DataFrame
        Combined results DataFrame.
    """
    result = dfs[0]
    for df in dfs[1:]:
        result = pd.merge(result, df, left_index=True, right_index=True)
    result.to_csv(os.path.join(results_to, filename), index=True)
    return result


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