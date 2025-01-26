import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

def normalize_dataframe(df: pd.DataFrame, feature_range: Tuple[float, float] = (0, 1)) -> pd.DataFrame:
    # Ensure input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Initialize the MinMaxScaler with the specified feature range
    scaler = MinMaxScaler(feature_range=feature_range)
    
    # Fit and transform the data
    normalized_data = scaler.fit_transform(df)
    
    # Convert the result back to a DataFrame, maintaining column names
    normalized_df = pd.DataFrame(normalized_data, index=df.index, columns=df.columns)
    
    return normalized_df