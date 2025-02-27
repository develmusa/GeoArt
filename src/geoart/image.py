
from typing import Optional
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator
import numpy as np
import base64

class Image(BaseModel):
    image_array: np.ndarray
    width: int
    height: int

    model_config = ConfigDict(arbitrary_types_allowed=True)


    def get_image(self, channels: int = 4) -> np.ndarray:
        """
        Reshape the image array using stored width, height, and optional channels.
        """
        return self.image_array.reshape((self.height, self.width, channels))

def create_image1(df: pd.DataFrame) -> Image:
    df = df[["temperature"]]

    normalized_data = (df - df.min().min()) / (df.max().max() - df.min().min())

    color_map = mpl.colormaps["viridis"](normalized_data.values)

    image_data = (color_map[:, :, :3] * 255).astype(np.uint8)
    plt.imshow(image_data)
    plt.axis('off')  # Hide the axis
    plt.show()
    

    



    return create_image(df.to_dataframe_byte_normalized())


def create_image(df: pd.DataFrame, color_map_str: str, min_temp: Optional[float] = None, max_temp: Optional[float] = None) -> Image:
    """
    Create an image from temperature data with optional custom temperature range.
    
    Args:
        df: DataFrame containing temperature data
        color_map_str: Name of the matplotlib colormap to use
        min_temp: Optional minimum temperature for color scaling (°C)
        max_temp: Optional maximum temperature for color scaling (°C)
        
    Returns:
        Image object containing the visualization
    """
    days = df["time"].dt.date.unique().size

    day_scaling_factor = 10

    image_height = days * day_scaling_factor
    image_width = image_height

    total_required_temperature_values = image_width * days

    df['time'] = pd.to_datetime(df['time'])

    new_time_range = pd.date_range(start=df['time'].iloc[0], end=df['time'].iloc[-1], periods=total_required_temperature_values)

    df_interpolated = pd.DataFrame({'time': new_time_range})
    df_interpolated['temperature'] = np.interp(
    df_interpolated['time'].astype(np.int64),
        df['time'].astype(np.int64),
        df['temperature']
    )

    # Reshape the array to match 'days' for repeating
    reshaped_data = df_interpolated['temperature'].fillna(0).to_numpy().reshape((days, image_width))
    
    # Repeat each day's data 3 times
    repeated_data = np.repeat(reshaped_data, day_scaling_factor, axis=0)
    
    # Use custom temperature range if provided, otherwise use data min/max
    data_min = min_temp if min_temp is not None else repeated_data.min()
    data_max = max_temp if max_temp is not None else repeated_data.max()
    
    # Ensure min is less than or equal to max
    if data_min > data_max:
        # Swap values if min > max
        data_min, data_max = data_max, data_min
    
    # Ensure min and max are different to avoid division by zero
    if data_min == data_max:
        data_max = data_min + 1.0
    
    # Normalize data using provided or calculated min/max values
    normalized_data = np.clip((repeated_data - data_min) / (data_max - data_min), 0, 1)
    
    color_map = mpl.colormaps[color_map_str](normalized_data) * 255
    color_map = color_map.flatten().astype(np.uint8)
    # Flatten the repeated data
    image = Image(image_array=color_map, width=image_width, height=image_height)
    return image    
