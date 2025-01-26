import pytest
import pandas as pd
from geoart.data_utils import normalize_dataframe

def test_normalize_dataframe_standard_range():
    # Test with default feature range (0, 1)
    df = pd.DataFrame({
        'A': [0, 1, 2, 3],
        'B': [100, 200, 300, 400]
    })
    expected_df = pd.DataFrame({
        'A': [0.0, 0.333333, 0.666667, 1.0],
        'B': [0.0, 0.333333, 0.666667, 1.0]
    })
    result = normalize_dataframe(df)
    pd.testing.assert_frame_equal(result, expected_df, rtol=1e-5)

def test_normalize_dataframe_custom_range():
    # Test with a custom feature range (-1, 1)
    df = pd.DataFrame({
        'A': [0, 1, 2, 3],
        'B': [100, 200, 300, 400]
    })
    expected_df = pd.DataFrame({
        'A': [-1.0, -0.333333, 0.333333, 1.0],
        'B': [-1.0, -0.333333, 0.333333, 1.0]
    })
    result = normalize_dataframe(df, feature_range=(-1, 1))
    pd.testing.assert_frame_equal(result, expected_df, rtol=1e-5)

def test_normalize_dataframe_single_column():
    # Test normalization of a single column DataFrame
    df = pd.DataFrame({
        'A': [1, 2, 3, 4]
    })
    expected_df = pd.DataFrame({
        'A': [0.0, 0.333333, 0.666667, 1.0]
    })
    result = normalize_dataframe(df)
    pd.testing.assert_frame_equal(result, expected_df, rtol=1e-5)

def test_normalize_empty_dataframe():
    # Test with an empty DataFrame
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        normalize_dataframe(df)

def test_input_not_dataframe():
    # Test input is not a DataFrame
    with pytest.raises(ValueError):
        normalize_dataframe([1, 2, 3])

def test_normalize_dataframe_nan_values():
    # Test handling of NaN values
    df = pd.DataFrame({
        'A': [1, 2, None, 4],
        'B': [None, 2, 3, 4]
    })
    # The NaN values should stay as NaN after normalization
    result = normalize_dataframe(df)
    assert result.isnull().equals(df.isnull())