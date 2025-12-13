import pytest
import sys
import os
import pandas as pd
import pandera.pandas as pa
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.validate_target_distribution import validate_target_distribution

# Valid distributions
# should return an equal dataframe as the input
perfect_balance_data = pd.DataFrame({
    'Washrooms': ['Y'] * 50 + ['N'] * 50
})
valid_data_other_col = pd.DataFrame({
    'Washrooms': ['Y'] * 50 + ['N'] * 50,
    'SpecialFeatures': ['Y'] * 45 + ['N'] * 55
})
valid_edge_case1 = pd.DataFrame({
    'Washrooms': ['Y'] * 80 + ['N'] * 20
})
valid_edge_case2 = pd.DataFrame({
    'Washrooms': ['Y'] * 20 + ['N'] * 80
})
def test_valid_data_passes():
    assert isinstance(validate_target_distribution(perfect_balance_data), pd.DataFrame)
    assert validate_target_distribution(perfect_balance_data).equals(perfect_balance_data)
    assert validate_target_distribution(valid_data_other_col).equals(valid_data_other_col)
    assert validate_target_distribution(valid_edge_case1).equals(valid_edge_case1)
    assert validate_target_distribution(valid_edge_case2).equals(valid_edge_case2)

# if there is an imbalanced distribution the function should raise a Schema Error
invalid_data1 = pd.DataFrame({
    'Washrooms': ['Y'] * 90 + ['N'] * 10
})
invalid_data2 = pd.DataFrame({
    'Washrooms': ['Y'] * 10 + ['N'] * 90
})
def test_invalid_data_errors():
    with pytest.raises(pa.errors.SchemaErrors):
        validate_target_distribution(invalid_data1)
    with pytest.raises(pa.errors.SchemaErrors):
        validate_target_distribution(invalid_data2)

# function should raise a TypeError if input is not a dataframe
data_not_df = perfect_balance_data.copy().to_numpy()
def test_valid_data_type():
    with pytest.raises(TypeError):
        validate_target_distribution(data_not_df)

# function should raise ValueError if input is empty
empty_data = pd.DataFrame({
    'Washrooms': []
})
def test_empty_error():
    with pytest.raises(ValueError):
        validate_target_distribution(empty_data)