import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize_dataframe_to_bytes(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Initialize the MinMaxScaler with the byte range (0-255)
    scaler = MinMaxScaler(feature_range=(0, 255))
    
    # Fit and transform the data
    normalized_data = scaler.fit_transform(df)
    
    # Convert the scaled data to integers within the byte range
    normalized_data = normalized_data.round().astype(int)
    
    # Convert the result back to a DataFrame, maintaining column names
    normalized_df = pd.DataFrame(normalized_data, index=df.index, columns=df.columns)
    
    return normalized_df