# eda.py
# author: Harry Yau
# date 2025-12-02

import pandas as pd
import click
import matplotlib.pyplot as plt
import os

@click.command()
@click.option('--train', type=str, help="Path to processed training data")
@click.option('--plot_to', type=str, help="Path to save plot")
def main(train, plot_to):
    '''Plots data analysis in the processed training data
        by class, display and save them'''
    train_df = pd.read_csv(train)
    train_df.describe().to_csv(f"{plot_to}/summary_stats.csv")
    # print(f"{plot_to}/summary_stats.csv")
    # pd.DataFrame(train_df.info()).to_csv(f"{plot_to}/column_info.csv")
    # Numeric feature
    numeric_features = 'Hectare'
    target = "Washrooms"
    # for col in numeric_features:
    train_df.groupby(target)[numeric_features].plot.hist(bins=50, alpha=0.5, legend=True)
    plt.xlabel(numeric_features);
    plt.savefig(os.path.join(plot_to, "numeric_feature.png"), dpi=300, bbox_inches="tight", transparent=False)

    # Categorical feature
    col = 'NeighbourhoodName'
    cat_df = train_df[[col, target]].copy()
    cat_df_count = cat_df.groupby([target, col]).size().unstack()
    cat_df_count.plot.bar()
    plt.title(col);
    plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left');
    plt.xlabel(col);
    plt.savefig(os.path.join(plot_to, "neightbourhood.png"), dpi=300, bbox_inches="tight", transparent=False)


    binary_features = ['Official', 'Advisories', 'SpecialFeatures', 'Facilities']
    
    # Visualize binary features when there is washroom
    binary_df = train_df[binary_features + [target]].copy()
    binary_df['Official'] = binary_df['Official'].astype(str).replace({"1": "Y", "0": "N"})
    washroom_df = binary_df[binary_df[target] == "Y"]
    washroom_df_count = washroom_df.groupby(target)[binary_features].apply(lambda group: (group == "Y").sum())
    washroom_df_count
    washroom_df_count.plot.bar()
    plt.title("Count of Binary Features when Washrooms are Present")
    plt.ylabel("Count")
    plt.savefig(os.path.join(plot_to, "binary_features_washroom.png"), dpi=300, bbox_inches="tight", transparent=False)

    # Visualize binary features when there is no washroom
    no_washroom_df = binary_df[binary_df[target] == "N"]
    no_washroom_df_count = no_washroom_df.groupby(target)[binary_features].apply(lambda group: (group == "Y").sum())
    no_washroom_df_count
    no_washroom_df_count.plot.bar()
    plt.title("Count of Binary Features when Washrooms are Absent")
    plt.ylabel("Count")
    plt.savefig(os.path.join(plot_to, "binary_features_nowashroom.png"), dpi=300, bbox_inches="tight", transparent=False)

    
if __name__ == '__main__':
    main()