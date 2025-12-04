# split_and_validate.py
# author: Rebecca Sokol-Snyder
# date: 2023-11-27

import os
import click
import json
import logging
import numpy as np
import pandas as pd
import pandera as pa
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@click.command()
@click.option('--raw-data', type=str, help="Path to raw data")
@click.option('--logs-to', type=str, help="Path to directory where validation logs will be written to")
@click.option('--data-to', type=str, help="Path to directory where processed data will be written to")
@click.option('--preprocessor-to', type=str, help="Path to directory where the preprocessor object will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
def main(raw_data, logs_to, data_to, preprocessor_to, seed):
    '''This script splits the raw data into train and test sets, 
    and then preprocesses the data to be used in exploratory data analysis.
    It also saves the preprocessor to be used in the model training script.'''
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

    park.to_csv(os.path.join(data_to, "parks_pandera_validated.csv"))
    
    # train and test data set up
    train_df, test_df = train_test_split(park, test_size=0.3, random_state=123)

    train_df.to_csv(os.path.join(data_to, "parks_train_raw.csv"), index=False)
    test_df.to_csv(os.path.join(data_to, "parks_test_raw.csv"), index=False)

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

    train_df.to_csv(os.path.join(data_to, "parks_train_target_validated.csv"), index=False)

    # categorizing features in parks
    numeric_features = ['Hectare']
    categorical_features = ['NeighbourhoodName']
    binary_features = ['Official', 'Advisories', 'SpecialFeatures', 'Facilities']
    drop_features = ['NeighbourhoodURL', 'ParkID', 'Name', 'GoogleMapDest', 'StreetNumber', 'StreetName', 'EWStreet', 'NSStreet']
    target = "Washrooms"


    # No anomalous correlations between target/response variable and features/explanatory variables
    # Using Deepchecks Feature Label Correlation check
    # Prepare dataset that matches Deepcheck syntax
    dc_categorical_features = ['NeighbourhoodName', 'Official', 'Advisories', 'SpecialFeatures', 'Facilities']
    dc_train_df = train_df.drop(columns = drop_features)
    dc_train_df.to_csv(os.path.join(data_to, "parks_train_for_deepchecks.csv"))

    # Checking procedure and result
    fl_check_ds = Dataset(dc_train_df, label=target, cat_features=dc_categorical_features)
    my_check = FeatureLabelCorrelation()
    feature_label = my_check.run(dataset=fl_check_ds)
    feature_la

    cancer_preprocessor = make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include='number')),
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    pickle.dump(cancer_preprocessor, open(os.path.join(preprocessor_to, "cancer_preprocessor.pickle"), "wb"))

    cancer_preprocessor.fit(cancer_train)
    scaled_cancer_train = cancer_preprocessor.transform(cancer_train)
    scaled_cancer_test = cancer_preprocessor.transform(cancer_test)

    scaled_cancer_train.to_csv(os.path.join(data_to, "scaled_cancer_train.csv"), index=False)
    scaled_cancer_test.to_csv(os.path.join(data_to, "scaled_cancer_test.csv"), index=False)

if __name__ == '__main__':
    main()