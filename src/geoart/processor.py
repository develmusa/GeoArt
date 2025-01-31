import asyncio
import base64
import datetime
from typing import Callable, Optional, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator
from geoart.data_utils import normalize_dataframe_to_bytes
import geoart.geolocation as geolocation
from geoart.geolocation import Coordinates
from geoart.image import Image
import geoart.weather_data as weather_data
from geoart.weather_data import WeatherData
import matplotlib.pyplot as plt
import matplotlib as mpl

class ProcessData(BaseModel):
    location_address: str
    location_coordinates: Optional[Coordinates] = None
    weather_data: Optional[WeatherData] = None
    start_date: datetime.date
    end_date: Optional[datetime.date] = None
    image: Optional[Image] = None
    # image: Optional[np.ndarray] = None  # To hold serialized NumPy array    image_array: Optional[np.ndarray] = None

class Notifier:
    def __init__(self, success_callback: Callable[[str, ProcessData], None], progress_callback: Optional[Callable[[str], None]] = None, error_callback: Optional[Callable[[str], None]] = None):
        self.success_callback = success_callback
        self.progress_callback = progress_callback
        self.error_callback = error_callback

    def notify_progress(self, message: str):
        if self.progress_callback:
            self.progress_callback(message)

    def notify_error(self, error_message: str):
        if self.error_callback:
            self.error_callback(error_message)

    def notify_success(self, message: str, process_data: ProcessData):
        self.success_callback(message, process_data)





def generate_year_temp_art(
    location_address: str,
    color_map: str,
    start_date: datetime.date,
    success_callback: Optional[Callable[[str, ProcessData], None]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    error_callback: Optional[Callable[[str], None]] = None,
) -> None:  # Define the exact return type if possible
    
    notifier = Notifier(success_callback=success_callback, progress_callback=progress_callback, error_callback=error_callback)
    process_data = ProcessData(location_address=location_address, start_date=start_date)
    process_data.end_date = start_date.replace(year=start_date.year + 1)

    async def async_wrapper():

        try:
            notifier.notify_progress("Fetching coordinates...")
            process_data.location_coordinates = geolocation.address_to_coordinates(address=location_address)
            notifier.notify_progress("Fetching weather data...")
            process_data.weather_data = await weather_data.fetch_weather_data(location_point=process_data.location_coordinates, start_date=process_data.start_date, end_date=process_data.end_date)
            # process_data.image = create_image(process_data.weather_data.hourly.to_dataframe_byte_normalized())
            process_data.image = create_image(process_data.weather_data.hourly.to_dataframe(), color_map)
            
            notifier.notify_success("Creating Art Complete", process_data)


            

        except geolocation.AddressNotFoundError:
            notifier.notify_error("Address not found")

        except weather_data.WeatherDataError:
            notifier.notify_error("Error fetching weather data")

        except Exception as e:
            notifier.notify_error(f"Unexpected Error {e}")
    asyncio.run(async_wrapper())

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
