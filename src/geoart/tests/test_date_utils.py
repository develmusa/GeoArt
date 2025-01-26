import pytest
import pandas as pd
from geoart.data_utils import normalize_dataframe_to_bytes

def test_normalize_dataframe_to_bytes_standard_case():
    # Test standard normalization case to bytes
    df = pd.DataFrame({
        'A': [0, 1, 2, 3],
        'B': [10, 20, 30, 40]
    })
    expected_df = pd.DataFrame({
        'A': [0, 85, 170, 255],
        'B': [0, 85, 170, 255]
    })
    result = normalize_dataframe_to_bytes(df)
    pd.testing.assert_frame_equal(result, expected_df)

def test_normalize_dataframe_to_bytes_single_column():
    # Test normalization of a single column DataFrame to bytes
    df = pd.DataFrame({
        'A': [1, 2, 3, 4]
    })
    expected_df = pd.DataFrame({
        'A': [0, 85, 170, 255]
    })
    result = normalize_dataframe_to_bytes(df)
    pd.testing.assert_frame_equal(result, expected_df)

def test_normalize_empty_dataframe_to_bytes():
    # Test with an empty DataFrame
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        normalize_dataframe_to_bytes(df)

def test_input_not_dataframe_to_bytes():
    # Test input is not a DataFrame
    with pytest.raises(ValueError):
        normalize_dataframe_to_bytes([1, 2, 3])

def test_normalize_dataframe_to_bytes_range_already_in_byte():
    # Test normalization where values are already in byte range
    df = pd.DataFrame({
        'A': [0, 64, 128, 255],
        'B': [10, 20, 30, 40]
    })
    expected_df = pd.DataFrame({
        'A': [0, 64, 128, 255],  # Limits preserved
        'B': [0, 85, 170, 255]
    })
    result = normalize_dataframe_to_bytes(df)
    pd.testing.assert_frame_equal(result, expected_df)
