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
@click.option('--train-data', type=str, help="Path to train data")
@click.option('--preprocessor', type=str, help="Path to parks_preprocessor.pickle")
#@click.option('--scaled-test-data', type=str, help="Path to scaled test data")
#@click.option('--y-data', type=str, help="Path to y data")
@click.option('--results-to', type=str, help="Path to directory where cross validation table will be written to")
@click.option('--pipeline-to', type=str, help="Path to directory where the model object will be written to")
#@click.option('--viz-to', type=str, help="Path to directory where visualizations will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
    
def main(train_data, preprocessor, results_to, pipeline_to, seed):
    '''Fits the parks washroom classifier on the training data 
    and saves the pipeline object results.'''
    np.random.seed(seed)
    set_config(transform_output="default")

    # Read Train Data and preprocessor
    target = "Washrooms"
    train_df = pd.read_csv(train_data)
    data_preprocessor = pickle.load(open(preprocessor, "rb"))

    # Split train data
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    # Dummy model implementation
    model = DummyClassifier(random_state=123)
    pipe_dummy = make_pipeline(data_preprocessor, model)
    dummy_df = pd.DataFrame({
        "dummy" : mean_std_cross_val_scores(pipe_dummy, X_train, y_train, cv=5, return_train_score=True)
    })
    
    # Write out result and pickle model
    dummy_df.transpose().to_csv(os.path.join(results_to, "dummy_result.csv"), index=True)
    with open(os.path.join(pipeline_to, "pipe_dummy_untrain.pickle"), 'wb') as f:
        pickle.dump(pipe_dummy, f)

    # RBF SVC model implementation
    svm_rbf_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')
    pipe_svm_rbf = make_pipeline(data_preprocessor, svm_rbf_classifier)
    svm_rbf_df = pd.DataFrame({
        "svm_rbf" : mean_std_cross_val_scores(pipe_svm_rbf, X_train, y_train, cv=5, return_train_score=True)
    })

    # Write out result and pickle model
    svm_rbf_df.transpose().to_csv(os.path.join(results_to, "svm_rbf_result.csv"), index=True)
    with open(os.path.join(pipeline_to, "pipe_svm_rbf_untrain.pickle"), 'wb') as f:
        pickle.dump(pipe_svm_rbf, f)

    # knn model implementation
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    pipe_knn = make_pipeline(data_preprocessor, knn_classifier)
    knn_df = pd.DataFrame({
        "knn" : mean_std_cross_val_scores(pipe_knn, X_train, y_train, cv=5, return_train_score=True)
    })

    # Write out result and pickle model
    knn_df.transpose().to_csv(os.path.join(results_to, "knn_result.csv"), index=True)
    with open(os.path.join(pipeline_to, "pipe_knn_untrain.pickle"), 'wb') as f:
        pickle.dump(pipe_knn, f)

    # Merge model Cross Validate results together
    result = pd.merge(dummy_df, svm_rbf_df, left_index=True, right_index=True)
    result = pd.merge(result, knn_df, left_index=True, right_index=True)
    result.to_csv(os.path.join(results_to, "combined_result.csv"), index=True)

    # Fit model
    pipe_svm_rbf_fit = pipe_svm_rbf.fit(X_train, y_train)
    with open(os.path.join(models_to, "pipe_svm_rbf_fully_trained.pickle"), 'wb') as f:
        pickle.dump(pipe_svm_rbf_fit, f)

if __name__ == '__main__':
    main()