
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


def create_image(df: pd.DataFrame, color_map_str: str) -> Image:

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
    normalized_data = (repeated_data - repeated_data.min()) / (repeated_data.max() - repeated_data.min())
    color_map = mpl.colormaps[color_map_str](normalized_data) * 255
    print()
    color_map = color_map.flatten().astype(np.uint8)
    # Flatten the repeated data
    image = Image(image_array=color_map, width=image_width, height=image_height)
    return image    
