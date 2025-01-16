import httpx
import pandas as pd
from pydantic import BaseModel, ValidationError, Field
from pydantic import BaseModel, ValidationError
from typing import List
from datetime import datetime

# Define Pydantic models for the API response
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
        pivoted_dataframe = (pivoted_dataframe - min_temp) / (max_temp - min_temp) * 100
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
        print(hourly_dataframe.max())

    asyncio.run(main())