import numpy as np
from PIL import Image
import httpx
import pandas as pd
from pydantic import BaseModel, ValidationError
from typing import List
from datetime import datetime

class HourlyData(BaseModel):
    time: List[str]
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

# # Function to process the hourly data into a Pandas DataFrame
# def process_hourly_data(weather_data: WeatherResponse):
#     hourly = weather_data.hourly
#     hourly_data = {
#         "date": pd.to_datetime(hourly.time),
#         "temperature_2m": hourly.temperature_2m,
#     }
#     hourly_dataframe = pd.DataFrame(data=hourly_data)
#     return hourly_dataframe


def process_hourly_data(weather_data: WeatherResponse):
    hourly = weather_data.hourly
    # Convert the time strings to datetime objects
    hourly_data = {
        "datetime": pd.to_datetime(hourly.time),
        "temperature_2m": hourly.temperature_2m,
    }
    hourly_dataframe = pd.DataFrame(data=hourly_data)

    # Create separate columns for date and time
    hourly_dataframe["date"] = hourly_dataframe["datetime"].dt.date
    hourly_dataframe["time"] = hourly_dataframe["datetime"].dt.time

    pivoted_dataframe = hourly_dataframe.pivot(index="date", columns="time", values="temperature_2m")

    min_temp = pivoted_dataframe.min().min()
    max_temp = pivoted_dataframe.max().max()

    if max_temp > min_temp:
        pivoted_dataframe = (pivoted_dataframe - min_temp) / (max_temp - min_temp) * 256
    else:
        pivoted_dataframe = pivoted_dataframe * 0

    return pivoted_dataframe

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

        # Flatten the pivoted DataFrame and handle NaN values (use zero for NaNs here)
        flattened_data = hourly_dataframe.fillna(0).to_numpy().flatten()

        # Specify desired dimensions for the image
        desired_width = 24  # Define as per your requirement
        desired_height = 360  # Define as per your requirement

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
        image_array = flattened_data.reshape((desired_height, desired_width)).astype(np.uint8)

        # Create and save the image
        image = Image.fromarray(image_array, mode='L')  # "L" for grayscale
        image.save("output_image_grayscale.png")
        print("Grayscale image saved as 'output_image_grayscale.png'")

    asyncio.run(main())