import pandas as pd
import pandera.pandas as pa

def validate_target_distribution(parks_training_df):
    """
    Validates that the training dataset has a reasonable distribution of target values (Washrooms and No Washrooms) without a major class imbalance that should be taken into consideration.

    Parameters
    ----------
    parks_training_df : pandas.DataFrame
        DataFrame containing Vancouver parks training data after the entire dataset has been split into training and test sets. 

    Returns
    -------
    pandas.DataFrame
        The validated DataFrame, confirmed to not have a class imbalance that should be taken into consideration

    Raises
    ------
    pandera.errors.SchemaError
        If the distribution of the target ('Washrooms') is too imbalanced, a Schema Error will be raised preventing continued analysis from occuring.
    
    Notes
    -----
    We expect both possible values of 'Washrooms' to occur in at least 20% of the rows of the training dataframe. We have also previously validated that all values in the 'Washrooms' column are not null and have either the value 'Y' or 'N'
    """
    if not isinstance(parks_training_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")    
    if parks_training_df.empty:
        raise ValueError("Dataframe must contain observations.")


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

    return training_schema.validate(parks_training_df, lazy=True)