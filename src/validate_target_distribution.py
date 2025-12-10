import pandas as pd
import pandera as pa

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
    We expect both possible values of 'Washrooms' to occur in at least 20% of the rows of the training dataframe
    """