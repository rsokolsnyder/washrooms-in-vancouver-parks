# split_and_validate.py
# author: Rebecca Sokol-Snyder
# date: 2023-11-27

import os
import click
import json
import logging
import numpy as np
import pandas as pd
import pandera.pandas as pa
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureLabelCorrelation
from deepchecks.tabular.checks.data_integrity import FeatureFeatureCorrelation


@click.command()
@click.option('--raw-data', type=str, help="Path to raw data")
@click.option('--logs-to', type=str, help="Path to directory where validation logs will be written to")
@click.option('--data-to', type=str, help="Path to directory where processed data will be written to")
@click.option('--viz-to', type=str, help="Path to directory where visualizations will be written to")
@click.option('--preprocessor-to', type=str, help="Path to directory where the preprocessor object will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
def main(raw_data, logs_to, data_to, viz_to, preprocessor_to, seed):
    '''This script splits the raw data into train and test sets, 
    performs data validation, and builds the preprocessor.'''
    np.random.seed(seed)

    park = pd.read_csv(raw_data, sep=';')
    
    # set up invalid data logging
    # adapted from DSCI 522 Textbook
    logging.basicConfig(
        filename=os.path.join(logs_to, "validation_errors.log"),
        filemode="w",
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
    )

    # Configure valid data schema
    schema = pa.DataFrameSchema(
        {
            "ParkID": pa.Column(int),
            "Name": pa.Column(str),
            "Official": pa.Column(int, pa.Check.isin([0,1])),
            "Advisories": pa.Column(str, pa.Check.isin(["Y", "N"])),
            "SpecialFeatures": pa.Column(str, pa.Check.isin(["Y", "N"])),
            "Facilities": pa.Column(str, pa.Check.isin(["Y", "N"])),
            "Washrooms": pa.Column(str, pa.Check.isin(["Y", "N"])),
            "StreetNumber": pa.Column(int, pa.Check.between(1, 10000), required=False),
            "StreetName": pa.Column(str, required=False),
            "EWStreet": pa.Column(str, nullable=True, required=False),
            "NSStreet": pa.Column(str, nullable=True, required=False),
            "NeighbourhoodName": pa.Column(str, pa.Check.isin([
                "Arbutus-Ridge",
                "Downtown",
                "Dunbar-Southlands",
                "Fairview",
                "Grandview-Woodland",
                "Hastings-Sunrise",
                "Kensington-Cedar Cottage",
                "Kerrisdale",
                "Killarney",
                "Kitsilano",
                "Mount Pleasant",
                "South Cambie",
                "Renfrew-Collingwood",
                "Oakridge",
                "Riley Park",
                "Shaughnessy",
                "Victoria-Fraserview",
                "West End",
                "West Point Grey",
                "Marpole",
                "Strathcona",
                "Sunset"
            ])),
            "NeighbourhoodURL": pa.Column(str, 
                checks=[
                    pa.Check(lambda url: url.str.startswith("https://vancouver.ca")),
                    pa.Check(lambda url: url.str.endswith(".aspx"))
                ],
                nullable=True
            ),
            "Hectare": pa.Column(float, pa.Check.between(0, 400)),
            "GoogleMapDest": pa.Column(str, 
                pa.Check(lambda latlon: latlon.str.startswith("49.")),
                nullable=True
            )
        },
        checks=[
            pa.Check(lambda df: ~df.duplicated().any(), error = "Duplicate Rows!"),
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error = "Empty Rows!")
        ],
        drop_invalid_rows=False
    )

    # Adapted from DSCI 522 Textbook
    # Initialize error cases DataFrame
    error_cases = pd.DataFrame()
    data = park.copy()

    # Adapted from DSCI 522 Textbook
    # Validate data and handle errors
    try:
        park = schema.validate(data, lazy=True)
    except pa.errors.SchemaErrors as e:
        error_cases = e.failure_cases

        # Adapted from DSCI 522 Textbook
        # Convert the error message to a JSON string
        error_message = json.dumps(e.message, indent=2)
        logging.error("\n" + error_message)

    # Filter out invalid rows based on the error cases
    if not error_cases.empty:
        invalid_indices = error_cases["index"].dropna().unique()
        park = (
            data.drop(index=invalid_indices)
            .reset_index(drop=True)
            .drop_duplicates()
            .dropna(how="all")
        )
    else:
        park = data

    park.to_csv(os.path.join(data_to, "pandera_validated_parks.csv"))
    
    # train and test data set up
    train_df, test_df = train_test_split(park, test_size=0.3, random_state=123)

    train_df.to_csv(os.path.join(data_to, "parks_train.csv"), index=False)
    test_df.to_csv(os.path.join(data_to, "parks_test.csv"), index=False)

    # Data Validation Pandera check of target distribution
    training_schema = pa.DataFrameSchema({
        "Washrooms": pa.Column(str, checks = [
            pa.Check(
                lambda w: (w == "Y").sum() / len(w) >= 0.2, 
                element_wise=False,
                error="Target Class may be imbalanced, check source data!"
            ),
            pa.Check(
                lambda w: (w == "N").sum() / len(w) >= 0.2, 
                element_wise=False,
                error="Target Class may be imbalanced, check source data!"
            )
        ])
    })

    train_df = training_schema.validate(train_df, lazy=True)

    train_df.to_csv(os.path.join(data_to, "parks_train.csv"), index=False)

    # categorizing features in parks
    numeric_features = ['Hectare']
    categorical_features = ['NeighbourhoodName']
    binary_features = ['Official', 'Advisories', 'SpecialFeatures', 'Facilities']
    drop_features = ['NeighbourhoodURL', 'ParkID', 'Name', 'GoogleMapDest', 'StreetNumber', 'StreetName', 'EWStreet', 'NSStreet']
    target = "Washrooms"


    # No anomalous correlations between target/response variable and features/explanatory variables
    # Using Deepchecks Feature Label Correlation check
    # Prepare dataset that matches Deepcheck syntax
    dc_categorical_features = categorical_features + binary_features
    dc_train_df = train_df.drop(columns = drop_features)
    dc_train_df.to_csv(os.path.join(data_to, "deepchecks_parks_train.csv"))

    # Checking procedure and result
    fl_check_ds = Dataset(dc_train_df, label=target, cat_features=dc_categorical_features)
    my_check = FeatureLabelCorrelation()
    feature_label_result = my_check.run(dataset=fl_check_ds)
    
    filename = os.path.join(viz_to, "feature_label_correlation.html")
    if os.path.exists(filename):
        os.remove(filename)

    feature_label_result.save_as_html(
        filename,
        as_widget=False
    )

    # No anomalous correlations between features/explanatory variables
    # Using Deepchecks Feature Feature Correlation check

    # Checking procedure and result
    ff_check_ds = Dataset(dc_train_df, cat_features=dc_categorical_features)
    check = FeatureFeatureCorrelation()
    check.add_condition_max_number_of_pairs_above_threshold(0.7, 3) # add self-defined threshold condition
    feature_feature_result = check.run(ff_check_ds)
    
    filename = os.path.join(viz_to, "feature_feature_correlation.html")
    if os.path.exists(filename):
        os.remove(filename)

    feature_feature_result.save_as_html(
        filename,
        as_widget=False
    )

    # preprocessor for column transformation
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"), OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )
    parks_preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(), binary_features),
        (categorical_transformer, categorical_features),
        ("drop", drop_features)
    )

    pickle.dump(parks_preprocessor, open(os.path.join(preprocessor_to, "parks_preprocessor.pickle"), "wb"))

    parks_preprocessor.fit(train_df)
    scaled_parks_train = pd.DataFrame(
        data=parks_preprocessor.transform(train_df),
        columns=parks_preprocessor.get_feature_names_out()
    )
    scaled_parks_test = pd.DataFrame(
        data=parks_preprocessor.transform(test_df),
        columns=parks_preprocessor.get_feature_names_out()
    )

    scaled_parks_train.to_csv(os.path.join(data_to, "scaled_parks_train.csv"), index=False)
    scaled_parks_test.to_csv(os.path.join(data_to, "scaled_parks_test.csv"), index=False)

if __name__ == '__main__':
    main()