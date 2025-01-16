import numpy as np
from PIL import Image
import httpx
import pandas as pd
from pydantic import BaseModel, ValidationError
from typing import Any, List
from datetime import datetime

class HourlyData(BaseModel):
    time: List[datetime]
    temperature_2m: List[float]

class WeatherResponse(BaseModel):
    latitude: float
    longitude: float
    elevation: float
    timezone: str
    timezone_abbreviation: str
    utc_offset_seconds: int
    hourly: HourlyData

# Function to fetch weather data
async def fetch_weather_data():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 47.3667,
        "longitude": 8.55,
        "start_date": "2023-12-31",
        "end_date": "2025-01-14",
        "hourly": "temperature_2m",
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()  # Raise for HTTP errors
        try:
            data = response.json()
            # Parse and validate the response using Pydantic
            weather_data = WeatherResponse(**data)
            return weather_data
        except ValidationError as e:
            print("Validation Error:", e)
            raise
        except Exception as e:
            print("Unexpected Error:", e)
            raise

from scipy.interpolate import make_splrep, splev


import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def interpolate_temperature(hourly_dataframe: pd.DataFrame, frequency="H"):
    # Resample to create regular intervals
    hourly_dataframe.set_index('datetime', inplace=True)
    
    # Create new regular time index with desired frequency
    new_index = pd.date_range(start=hourly_dataframe.index.min(),
                              end=hourly_dataframe.index.max(),
                              freq=frequency)
    
    # Reindex with the new regular index
    interpolated_df = hourly_dataframe.reindex(new_index)

    # Linearly interpolate missing values
    interpolated_temperatures = interp1d(
        x=hourly_dataframe.index.values.astype(np.int64), 
        y=hourly_dataframe.temperature_2m.values, 
        bounds_error=False,
        fill_value="extrapolate")(interpolated_df.index.values.astype(np.int64))
    
    interpolated_df["temperature_2m"] = interpolated_temperatures
    
    # Reset index
    interpolated_df.reset_index(inplace=True)
    interpolated_df.rename(columns={'index': 'datetime'}, inplace=True)
    
    return interpolated_df

# Example usage with original DataFrame and desired number of points
# hourly_dataframe = ...  # Your original DataFrame
# interpolated_df = interpolate_temperature(hourly_dataframe, num_points=20000)
def process_hourly_data(weather_data: WeatherResponse):
    hourly = weather_data.hourly
    # Convert the time strings to datetime objects
    hourly_data = {
        "datetime": pd.to_datetime(hourly.time),
        "temperature_2m": hourly.temperature_2m,
    }
    hourly_dataframe = pd.DataFrame(data=hourly_data)

    print(hourly_dataframe.info())

    hourly_dataframe = interpolate_temperature(hourly_dataframe, "min")

    # Create separate columns for date and time
    hourly_dataframe["date"] = hourly_dataframe["datetime"].dt.date
    hourly_dataframe["time"] = hourly_dataframe["datetime"].dt.time

    pivoted_dataframe = hourly_dataframe.pivot(index="date", columns="time", values="temperature_2m")

    min_temp = pivoted_dataframe.min().min()
    max_temp = pivoted_dataframe.max().max()

    if max_temp > min_temp:
        pivoted_dataframe = (pivoted_dataframe - min_temp) / (max_temp - min_temp) * 255
    else:
        pivoted_dataframe = pivoted_dataframe * 0

    return pivoted_dataframe

def original(hourly_dataframe: pd.DataFrame) -> np.ndarray[Any]:

    # Flatten the pivoted DataFrame and handle NaN values (use zero for NaNs here)
    flattened_data = hourly_dataframe.fillna(0).to_numpy().flatten()

    # Specify desired dimensions for the image
    desired_width = hourly_dataframe.shape[1]  # Define as per your requirement
    desired_height = hourly_dataframe.shape[0]  # Define as per your requirement

    # Ensure the data fits into the specified dimensions
    total_pixels = desired_width * desired_height
    flattened_data_size = len(flattened_data)

    if flattened_data_size < total_pixels:
        # If there are not enough data points, pad with zeros
        flattened_data = np.append(flattened_data, np.zeros(total_pixels - flattened_data_size))
    elif flattened_data_size > total_pixels:
        # If there are too many data points, truncate the array
        flattened_data = flattened_data[:total_pixels]

    # Reshape and convert to a NumPy array of the correct type
    return flattened_data.reshape((desired_height, desired_width)).astype(np.uint8)


if __name__ == "__main__":
    import asyncio

    async def main():
        weather_data = await fetch_weather_data()
        print(f"Coordinates {weather_data.latitude}°N {weather_data.longitude}°E")
        print(f"Elevation {weather_data.elevation} m asl")
        print(f"Timezone {weather_data.timezone} {weather_data.timezone_abbreviation}")
        print(f"Timezone difference to GMT+0 {weather_data.utc_offset_seconds} s")

        hourly_dataframe = process_hourly_data(weather_data)
        print(hourly_dataframe)

        image_array = original(hourly_dataframe)

        # Create and save the image
        image = Image.fromarray(image_array, mode='L')  # "L" for grayscale
        image.save("output_image_grayscale.png")
        print("Grayscale image saved as 'output_image_grayscale.png'")

    asyncio.run(main())