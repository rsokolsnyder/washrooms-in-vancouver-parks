# build_model_to_predict_washroom.py
# author: William Song
# date: 2025-12-02

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
@click.option('--scaled-train-data', type=str, help="Path to scaled train data")
@click.option('--scaled-test-data', type=str, help="Path to scaled test data")
@click.option('--y-data', type=str, help="Path to y data")
@click.option('--results-to', type=str, help="Path to directory where cross validation table will be written to")
@click.option('--pipeline-to', type=str, help="Path to directory where the model object will be written to")
@click.option('--viz-to', type=str, help="Path to directory where visualizations will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
def main(scaled_train_data, scaled_test_data, y_data, results_to, pipeline_to, viz_to, seed):
    '''Evaluates the breast cancer classifier on the test data 
    and saves the evaluation results.'''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    # Read Scaled Train Test Data and Split to X, y
    target = "Washrooms"
    
    train_df = pd.read_csv(scaled_train_data)
    test_df = pd.read_csv(scaled_test_data)

    X_train = train_df
    y_train = pd.read_csv(y_data)
    #X_train = train_df.drop(columns=[target])
    #y_train = train_df[target]
    #X_test = test_df.drop(columns=[target])
    #y_test = test_df[target]

    # Dummy model implementation
    model = DummyClassifier(random_state=123)
    pipe_dummy = make_pipeline(model)

    dummy_df = pd.DataFrame({
        "dummy" : mean_std_cross_val_scores(pipe_dummy, X_train, y_train, cv=5, return_train_score=True)
    })
    dummy_df.transpose().to_csv(os.path.join(results_to, "dummy_result.csv"), index=False)
    with open(os.path.join(pipeline_to, "pipe_dummy_untrain.pickle"), 'wb') as f:
        pickle.dump(pipe_dummy, f)

    # RBF SVC model implementation
    svm_rbf_classifier = SVC(kernel='rbf', C=1.0, gamma='scale') 
    pipe_svm_rbf = make_pipeline(svm_rbf_classifier)
    svm_rbf_df = pd.DataFrame({
        "svm_rbf" : mean_std_cross_val_scores(pipe_svm_rbf, X_train, y_train, cv=5, return_train_score=True)
    })
    svm_rbf_df.transpose().to_csv(os.path.join(results_to, "svm_rbf_result.csv"), index=False)
    with open(os.path.join(pipeline_to, "pipe_svm_rbf_untrain.pickle"), 'wb') as f:
        pickle.dump(pipe_svm_rbf, f)

    # knn model implementation
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    pipe_knn = make_pipeline(knn_classifier)
    knn_df = pd.DataFrame({
        "knn" : mean_std_cross_val_scores(pipe_knn, X_train, y_train, cv=5, return_train_score=True)
    })
    knn_df.transpose().to_csv(os.path.join(results_to, "knn_result.csv"), index=False)
    with open(os.path.join(pipeline_to, "pipe_knn_untrain.pickle"), 'wb') as f:
        pickle.dump(pipe_knn, f)

    # Merge model Cross Validate results together
    result = pd.merge(dummy_df, svm_rbf_df, left_index=True, right_index=True)
    result = pd.merge(result, knn_df, left_index=True, right_index=True)
    result.to_csv(os.path.join(results_to, "combined_result.csv"), index=False)

    # Fit model
    pipe_svm_rbf_fit = pipe_svm_rbf.fit(X_train, y_train)
    with open(os.path.join(pipeline_to, "pipe_svm_rbf_trained.pickle"), 'wb') as f:
        pickle.dump(pipe_svm_rbf_fit, f)

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