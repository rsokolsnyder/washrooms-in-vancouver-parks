# eda.py
# author: Harry Yau
# date 2025-12-02

import pandas as pd
import click
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from eda_utils import save_summary_statistics, plot_numeric_feature, plot_categorical_feature, plot_binary_features

@click.command()
@click.option('--training-data', type=str, help="Path to processed training data")
@click.option('--plot-to', type=str, help="Path to save plot")
@click.option('--tables-to', type=str, help="Path to save plot")
def main(training_data, tables_to, plot_to):
    '''Plots data analysis in the processed training data
        by class, display and save them'''
    train_df = pd.read_csv(training_data)

    # Save summary statistics
    save_summary_statistics(train_df, tables_to)

    # Numeric feature
    plot_numeric_feature(train_df, numeric_col="Hectare", target="Washrooms", plot_to=plot_to)

    # Categorical feature
    plot_categorical_feature(train_df, cat_col="NeighbourhoodName", target="Washrooms", plot_to=plot_to)

    # Binary features
    binary_features = ['Official', 'Advisories', 'SpecialFeatures', 'Facilities']
    plot_binary_features(train_df, binary_cols=binary_features, target="Washrooms", plot_to=plot_to)

    
if __name__ == '__main__':
    main()