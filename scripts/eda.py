# eda.py
# author: Harry Yau
# date 2025-12-02

import pandas as pd
import click
import matplotlib.pyplot as plt
import os

def save_summary_statistics(df, tables_to):
    """Save summary statistics to CSV."""
    summary_path = os.path.join(tables_to, "eda_summary_stats.csv")
    df.describe().to_csv(summary_path, index=False)
    return summary_path


def plot_numeric_feature(df, numeric_col, target, plot_to):
    """Plot histogram of a numeric feature grouped by class."""
    df.groupby(target)[numeric_col].plot.hist(bins=50, alpha=0.5, legend=True)
    plt.xlabel(numeric_col)
    out_path = os.path.join(plot_to, "numeric_feature.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=False)
    plt.clf()
    return out_path

def plot_categorical_feature(df, cat_col, target, plot_to):
    """Plot bar chart for a categorical feature by class."""
    cat_df = df[[cat_col, target]].copy()
    cat_counts = cat_df.groupby([target, cat_col]).size().unstack()
    cat_counts.plot.bar()

    plt.title(cat_col)
    plt.legend(title=cat_col, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xlabel(cat_col)

    out_path = os.path.join(plot_to, "neighbourhood.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=False)
    plt.clf()
    return out_path

def plot_binary_features(df, binary_cols, target, plot_to):
    """Plot binary feature counts separately for washroom=Y and N."""
    # Convert to Y/N strings
    binary_df = df[binary_cols + [target]].copy()
    binary_df["Official"] = binary_df["Official"].astype(str).replace({"1": "Y", "0": "N"})

    # Washrooms present
    washroom_df = binary_df[binary_df[target] == "Y"]
    washroom_counts = washroom_df.groupby(target)[binary_cols].apply(lambda g: (g == "Y").sum())
    washroom_counts.plot.bar()
    plt.title("Count of Binary Features when Washrooms are Present")
    plt.ylabel("Count")

    out_present = os.path.join(plot_to, "binary_features_washroom.png")
    plt.savefig(out_present, dpi=300, bbox_inches="tight", transparent=False)
    plt.clf()

    # No washrooms
    no_washroom_df = binary_df[binary_df[target] == "N"]
    no_counts = no_washroom_df.groupby(target)[binary_cols].apply(lambda g: (g == "Y").sum())
    no_counts.plot.bar()
    plt.title("Count of Binary Features when Washrooms are Absent")
    plt.ylabel("Count")

    out_absent = os.path.join(plot_to, "binary_features_nowashroom.png")
    plt.savefig(out_absent, dpi=300, bbox_inches="tight", transparent=False)
    plt.clf()

    return out_present, out_absent

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